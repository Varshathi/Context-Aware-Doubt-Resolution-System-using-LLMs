import { useState, useRef, useEffect, useCallback } from "react";

// ══════════════════════════════════════════════════════════════════════════════
//  SUMMARY STORE  — plain array, no vectors, no TF-IDF
//  Stores one-sentence summaries of every exchange.
//  Used as context in the Judge and Bridge prompts.
// ══════════════════════════════════════════════════════════════════════════════
class SummaryStore {
  constructor() { this.entries = []; }

  add(entry) { this.entries.push(entry); } // { turn, summary, chunkIndex, isDrift }

  // Returns the N most recent entries as a readable string for prompt injection
  getContext(n = 5) {
    return this.entries.slice(-n)
      .map(e => `Turn ${e.turn} [chunk ${e.chunkIndex ?? "?"}]${e.isDrift ? " [BRIDGE]" : ""}: ${e.summary}`)
      .join("\n");
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  PROMPTS
// ══════════════════════════════════════════════════════════════════════════════

const TEACHING_SYSTEM = `You are Socra — a warm, Socratic AI tutor. You guide students to discover knowledge themselves.

## CORE TEACHING RULES
- Decompose topics into 4–7 atomic chunks. Teach ONE chunk at a time.
- Max 3–4 sentences of explanation, then ask ONE Socratic question.
- Never give the final answer. Use hints, examples, analogies.
- Adapt: correct → advance | partial → hint | wrong → example | confused → simpler analogy

## CONNECTING THREAD — apply this to EVERY response, always
Every response must end with a natural thread back to the root topic and current chunk.
This is your default conversational style, not a special emergency mode.

Weave it in naturally depending on context:
- After answering a doubt: "...and that's exactly why [current chunk] works this way. So — [Socratic question about chunk]"
- After an analogy: "...which mirrors what happens in [root topic]. With that in mind, [Socratic question]"
- After encouragement: "...you're building the right mental model for [current chunk]. Let's go deeper — [question]"
- After a side concept: "...this connects directly to what we're exploring about [chunk]. So — [question]"

The thread should feel like one flowing conversation, never a jarring redirect.

## DECISION ACTIONS
introduce | socratic_check | hint | re_explain | example | advance | context_bridge

Use context_bridge when the student asked something clearly off-topic, you answered it briefly, and are now reconnecting back to the learning path.

## OUTPUT FORMAT — always include both blocks at the end of your response:

<knowledge_state>
{
  "topic": "...",
  "chunks": ["chunk1", "chunk2", "..."],
  "current_chunk_index": 0,
  "chunk_scores": { "chunk1": 0 },
  "overall_confidence": 0,
  "action_taken": "introduce"
}
</knowledge_state>

<exchange_summary>
One sentence (max 20 words) summarising what was taught or discussed this turn.
</exchange_summary>`;

// ── JUDGE: tiny call, returns JSON only ──────────────────────────────────────
// Evaluates the BOT'S OWN response — not the student's message.
// This means related doubts, clarifications, and hints all score correctly
// because we're checking whether SOCRA stayed on topic, not whether the
// student's casual message matched academic vocabulary.
const JUDGE_SYSTEM = `You are a strict learning-path monitor.

You will be given:
- ROOT TOPIC: the subject being taught
- CHUNKS: the planned learning sequence
- CURRENT CHUNK: where teaching is right now
- RECENT SUMMARIES: what has been covered so far
- BOT RESPONSE: the tutor's latest response to evaluate

Your job: decide if the BOT RESPONSE is still serving the learning path for the current chunk and topic, OR if it has genuinely drifted into unrelated territory.

IMPORTANT RULES:
- Related doubts, clarifications, analogies, and examples about the SAME topic = NOT drift
- Answering a student's side question that connects back = NOT drift  
- Explaining a prerequisite concept needed to understand the current chunk = NOT drift
- ONLY flag drift if the bot is teaching a completely unrelated subject

Respond with ONLY valid JSON, no other text:
{"drifted": false, "reason": "one sentence explanation"}
or
{"drifted": true, "reason": "one sentence explanation", "last_chunk": "name of chunk we drifted away from"}`;

// ── BRIDGE: reconnects from drifted response back to learning path ───────────
const BRIDGE_SYSTEM = `You are Socra — a Socratic tutor who has just noticed you went off-track.

You will receive:
- The student's original message
- The drifted response you generated (which should be discarded)  
- The root topic and current chunk you should be teaching
- Recent conversation summaries for context

Your task — write a NEW response that:
1. If the student asked a genuine question: answer it briefly (1–2 sentences)
2. Explicitly name the connection back to the root topic ("This actually connects to what we were exploring about X...")
3. Resume teaching the current chunk with a Socratic question

The bridge must feel like a natural segue, not a hard redirect. Be warm.

Always include the knowledge_state and exchange_summary blocks — keep current_chunk_index and chunk_scores UNCHANGED from what you're given, set action_taken to "context_bridge".

<knowledge_state>
{ ...same structure as teaching system... }
</knowledge_state>

<exchange_summary>
One sentence: briefly answered [topic], bridged back to [chunk].
</exchange_summary>`;

const SUMMARIZER_SYSTEM = `Extract the key learning point from this tutoring exchange in ONE sentence, max 20 words. Output only the sentence, no quotes or preamble.`;

// ══════════════════════════════════════════════════════════════════════════════
//  API HELPER
// ══════════════════════════════════════════════════════════════════════════════
async function callClaude(system, messages, maxTokens = 900) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: maxTokens,
      system,
      messages,
    }),
  });
  const data = await res.json();
  return data.content?.map(b => b.text || "").join("") || "";
}

// ══════════════════════════════════════════════════════════════════════════════
//  PARSE HELPERS
// ══════════════════════════════════════════════════════════════════════════════
function parseTag(text, tag) {
  const m = text.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`));
  return m ? m[1].trim() : null;
}

function stripTags(text) {
  return text
    .replace(/<knowledge_state>[\s\S]*?<\/knowledge_state>/g, "")
    .replace(/<exchange_summary>[\s\S]*?<\/exchange_summary>/g, "")
    .trim();
}

function parseKS(text) {
  try { return JSON.parse(parseTag(text, "knowledge_state") || "null"); }
  catch { return null; }
}

function parseJudge(text) {
  try {
    const clean = text.replace(/```json|```/g, "").trim();
    return JSON.parse(clean);
  } catch { return { drifted: false, reason: "parse error" }; }
}

// ══════════════════════════════════════════════════════════════════════════════
//  UI COMPONENTS
// ══════════════════════════════════════════════════════════════════════════════
function ScoreBar({ label, score }) {
  const color = score >= 80 ? "#4DFFA0" : score >= 50 ? "#FFD166" : score >= 20 ? "#FF9A3C" : "#1e293b";
  return (
    <div style={{ marginBottom: 9 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
        <span style={{ fontSize: 10.5, color: "#64748b", fontFamily: "'DM Mono',monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "72%" }}>{label}</span>
        <span style={{ fontSize: 10.5, color, fontFamily: "'DM Mono',monospace", fontWeight: 600 }}>{score}%</span>
      </div>
      <div style={{ height: 3, background: "#0f172a", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${score}%`, background: color, borderRadius: 2, transition: "width 0.7s cubic-bezier(0.34,1.56,0.64,1)", boxShadow: score > 0 ? `0 0 8px ${color}80` : "none" }} />
      </div>
    </div>
  );
}

function JudgeVerdict({ verdict }) {
  if (!verdict) return null;
  const color = verdict.drifted ? "#FF4D6D" : "#4DFFA0";
  const icon = verdict.drifted ? "⚠️" : "✓";
  const label = verdict.drifted ? "DRIFT DETECTED" : "ON TRACK";
  return (
    <div style={{ padding: "8px 10px", borderRadius: 8, background: `${color}0D`, border: `1px solid ${color}30`, marginBottom: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: verdict.reason ? 4 : 0 }}>
        <span style={{ fontSize: 11 }}>{icon}</span>
        <span style={{ fontSize: 9.5, color, fontFamily: "'DM Mono',monospace", fontWeight: 700, letterSpacing: "0.08em" }}>{label}</span>
      </div>
      {verdict.reason && (
        <div style={{ fontSize: 10.5, color: "#475569", lineHeight: 1.4, fontFamily: "'Lora',serif" }}>{verdict.reason}</div>
      )}
    </div>
  );
}

function MessageBubble({ msg, isNew }) {
  const isUser = msg.role === "user";
  const isBridge = msg.drift === true;

  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 20, animation: isNew ? "fadeIn 0.3s ease" : "none" }}>
      {!isUser && (
        <div style={{ width: 34, height: 34, borderRadius: "50%", background: isBridge ? "linear-gradient(135deg,#FF4D6D,#FFD166)" : "linear-gradient(135deg,#4DFFA0,#0EA5E9)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, flexShrink: 0, marginRight: 10, marginTop: 2, boxShadow: isBridge ? "0 0 14px #FF4D6D50" : "0 0 14px #4DFFA050" }}>
          {isBridge ? "🌉" : "🦉"}
        </div>
      )}
      <div style={{ maxWidth: "74%" }}>
        {isBridge && (
          <div style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 9px", borderRadius: 12, background: "rgba(255,77,109,0.1)", border: "1px solid rgba(255,77,109,0.3)", marginBottom: 6 }}>
            <span style={{ fontSize: 9, color: "#FF4D6D", fontFamily: "'DM Mono',monospace", letterSpacing: "0.08em", fontWeight: 700 }}>🌉 CONTEXT BRIDGE</span>
          </div>
        )}
        <div
          style={{ padding: "13px 16px", borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px", background: isUser ? "linear-gradient(135deg,#0EA5E9,#6366F1)" : isBridge ? "rgba(255,77,109,0.07)" : "rgba(15,24,42,0.95)", border: isUser ? "none" : isBridge ? "1px solid rgba(255,77,109,0.18)" : "1px solid rgba(77,255,160,0.07)", boxShadow: isUser ? "0 4px 20px rgba(14,165,233,0.2)" : "0 4px 16px rgba(0,0,0,0.3)", fontSize: 14.5, lineHeight: 1.72, color: isUser ? "#fff" : "#cbd5e1", fontFamily: "'Lora',Georgia,serif", whiteSpace: "pre-wrap" }}
          dangerouslySetInnerHTML={{
            __html: msg.content
              .replace(/\*\*(.*?)\*\*/g, "<strong style='color:#4DFFA0'>$1</strong>")
              .replace(/\*(.*?)\*/g, "<em style='color:#FFD166'>$1</em>")
              .replace(/`(.*?)`/g, "<code style='background:#0f172a;padding:2px 6px;border-radius:3px;font-family:DM Mono,monospace;font-size:12.5px;color:#4DFFA0'>$1</code>")
          }}
        />
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN APP
// ══════════════════════════════════════════════════════════════════════════════
const SUGGESTED = ["Quantum Entanglement", "Neural Networks", "CRISPR Gene Editing", "Black Holes", "Recursion in Programming", "Supply & Demand"];

// Teaching is considered started after turn index 2
// Turn 0: student names topic  → bot gives chunk preview + level question  [ONBOARDING]
// Turn 1: student answers level → bot begins chunk 1                        [TEACHING STARTS]
// Turn 2+: judge fires on every bot response
const JUDGE_STARTS_AT_TURN = 2;

export default function StudyModeV3() {
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [knowledge, setKnowledge]   = useState(null);
  const [rootQuery, setRootQuery]   = useState(null);
  const [lastVerdict, setLastVerdict] = useState(null);
  const [driftCount, setDriftCount] = useState(0);
  const [summaries, setSummaries]   = useState([]);
  const [statusMsg, setStatusMsg]   = useState("");
  const [showConfirm, setShowConfirm] = useState(false);
  const [turnIndex, setTurnIndex]   = useState(0); // counts completed student turns

  const bottomRef  = useRef(null);
  const inputRef   = useRef(null);
  const historyRef = useRef([]);   // full conversation for Claude
  const storeRef   = useRef(new SummaryStore());
  const ksRef      = useRef(null); // last known knowledge state

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  // ── NEW SESSION ─────────────────────────────────────────────────────────────
  const newSession = useCallback(() => {
    setMessages([]); setInput(""); setLoading(false);
    setKnowledge(null); setRootQuery(null); setLastVerdict(null);
    setDriftCount(0); setSummaries([]); setStatusMsg("");
    setShowConfirm(false); setTurnIndex(0);
    historyRef.current = [];
    storeRef.current = new SummaryStore();
    ksRef.current = null;
    setTimeout(() => inputRef.current?.focus(), 100);
  }, []);

  // ── SEND MESSAGE ─────────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (userText) => {
    if (!userText.trim() || loading) return;

    const currentTurn = turnIndex;
    const isFirst = currentTurn === 0;

    setInput("");
    setLoading(true);
    setLastVerdict(null);

    const nextMessages = [...messages, { role: "user", content: userText.trim() }];
    setMessages(nextMessages);
    setNewIdx_internal(nextMessages.length - 1);

    historyRef.current = [...historyRef.current, { role: "user", content: userText.trim() }];

    try {
      // ── STEP 1: Always generate the teaching response first ────────────────
      if (isFirst) {
        setRootQuery(userText.trim());
        setStatusMsg("📚 Building your learning path...");
      } else {
        setStatusMsg("🎓 Generating response...");
      }

      // Inject a context reminder into the system so Claude always knows
      // the root topic, current chunk, and recent session log.
      // This is what makes the connecting thread possible on every response.
      const currentKS = ksRef.current;
      const recentContext = storeRef.current.getContext(4);
      const contextInjection = isFirst ? "" : `

## CURRENT SESSION STATE
Root topic: "${rootQuery}"
Current chunk: "${currentKS?.chunks?.[currentKS?.current_chunk_index] || "not started yet"}" (index ${currentKS?.current_chunk_index ?? 0})
All chunks: ${JSON.stringify(currentKS?.chunks || [])}
Recent session log:
${recentContext || "(none yet)"}

Always end your response by connecting back to the current chunk listed above.`;

      const rawResponse = await callClaude(
        TEACHING_SYSTEM + contextInjection,
        historyRef.current
      );

      // ── STEP 2: Judge the bot's response — but only after onboarding ───────
      // We evaluate SOCRA'S output, not the student's input.
      // This means:
      //   - "I'm a beginner" from student → never evaluated (it's onboarding)
      //   - "what's a wave function?" (related doubt) → SOCRA's answer is judged
      //     and will correctly pass because it's still teaching the topic
      //   - "what's the French Revolution?" tangent → SOCRA's answer is judged
      //     and will correctly fail because SOCRA drifted into history
      let finalResponse = rawResponse;
      let isDrift = false;
      let judgeVerdict = null;

      if (currentTurn >= JUDGE_STARTS_AT_TURN) {
        setStatusMsg("🔍 Checking learning path...");

        const botResponseClean = stripTags(rawResponse);

        const judgePrompt = `ROOT TOPIC: "${rootQuery}"
PLANNED CHUNKS: ${JSON.stringify(currentKS?.chunks || [])}
CURRENT CHUNK: "${currentKS?.chunks?.[currentKS?.current_chunk_index] || "unknown"}" (index ${currentKS?.current_chunk_index ?? 0})

RECENT CONVERSATION SUMMARIES:
${recentContext || "(none yet)"}

BOT RESPONSE TO EVALUATE:
"${botResponseClean.slice(0, 600)}"`;

        const judgeRaw = await callClaude(JUDGE_SYSTEM, [{ role: "user", content: judgePrompt }], 150);
        judgeVerdict = parseJudge(judgeRaw);
        setLastVerdict(judgeVerdict);

        if (judgeVerdict?.drifted) {
          // ── STEP 3: Regenerate with bridge prompt ──────────────────────────
          isDrift = true;
          setDriftCount(c => c + 1);
          setStatusMsg("🌉 Building context bridge...");
          const bridgeUserMsg = `STUDENT SAID: "${userText}"

YOUR DRIFTED RESPONSE (discard this):
"${botResponseClean.slice(0, 400)}"

JUDGE VERDICT: ${judgeVerdict.reason}

ROOT TOPIC: "${rootQuery}"
CURRENT CHUNK TO RETURN TO: "${currentKS?.chunks?.[currentKS?.current_chunk_index] || "the main topic"}" (index ${currentKS?.current_chunk_index ?? 0})
ALL CHUNKS: ${JSON.stringify(currentKS?.chunks || [])}
CURRENT KNOWLEDGE STATE: ${JSON.stringify(currentKS)}

RECENT SUMMARIES FOR CONTEXT:
${recentContext || "(none yet)"}

Write a new response that bridges back to the current chunk.`;

          finalResponse = await callClaude(BRIDGE_SYSTEM, [
            ...historyRef.current.slice(0, -1),
            { role: "user", content: bridgeUserMsg }
          ]);
        }
      }

      // ── STEP 4: Parse knowledge state & summary ────────────────────────────
      const ks = parseKS(finalResponse);
      const exchangeSummaryRaw = parseTag(finalResponse, "exchange_summary");
      const cleanText = stripTags(finalResponse);

      if (ks) { setKnowledge(ks); ksRef.current = ks; }

      // Get a summary — use embedded tag first, else ask Claude
      let summaryText = exchangeSummaryRaw?.trim();
      if (!summaryText) {
        try {
          summaryText = await callClaude(SUMMARIZER_SYSTEM, [{
            role: "user",
            content: `Student: "${userText}"\nTutor: "${cleanText.slice(0, 300)}"`
          }], 60);
          summaryText = summaryText.trim().replace(/^["']|["']$/g, "");
        } catch { summaryText = userText.slice(0, 50); }
      }

      storeRef.current.add({
        turn: currentTurn,
        summary: summaryText,
        chunkIndex: ks?.current_chunk_index ?? null,
        isDrift,
      });
      setSummaries(prev => [...prev, { text: summaryText, isDrift, turn: currentTurn }]);

      // ── STEP 5: Commit to conversation ────────────────────────────────────
      historyRef.current = [...historyRef.current, { role: "assistant", content: finalResponse }];

      setMessages(prev => {
        const updated = [...prev, { role: "assistant", content: cleanText, drift: isDrift }];
        setNewIdx_internal(updated.length - 1);
        return updated;
      });

      setTurnIndex(t => t + 1);

    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: "assistant", content: "⚠️ Something went wrong. Please try again.", drift: false }]);
    } finally {
      setLoading(false);
      setStatusMsg("");
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [messages, loading, rootQuery, turnIndex]);

  // Internal new-message index tracker (avoids stale closure issues)
  const [newMsgIdx, setNewMsgIdx] = useState(-1);
  function setNewIdx_internal(i) { setNewMsgIdx(i); }

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(input); }
  };

  const actionColors = {
    introduce: "#4DFFA0", socratic_check: "#FFD166", hint: "#FF9A3C",
    re_explain: "#60A5FA", example: "#C084FC", advance: "#4DFFA0", context_bridge: "#FF4D6D"
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        body{background:#020817;}
        @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
        ::-webkit-scrollbar{width:3px}
        ::-webkit-scrollbar-thumb{background:#1e293b;border-radius:2px}
        textarea:focus{outline:none}
      `}</style>

      <div style={{ display: "flex", height: "100vh", background: "#020817", overflow: "hidden", fontFamily: "'Lora',serif" }}>

        {/* ── SIDEBAR ─────────────────────────────────────────────────────── */}
        <div style={{ width: 272, background: "rgba(5,11,22,0.98)", borderRight: "1px solid rgba(77,255,160,0.06)", display: "flex", flexDirection: "column", padding: "20px 14px", flexShrink: 0, overflowY: "auto" }}>

          {/* Logo + New Session */}
          <div style={{ marginBottom: 24 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 24, fontWeight: 800, background: "linear-gradient(135deg,#4DFFA0,#0EA5E9)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Socra v3</div>
              {messages.length > 0 && (
                <button onClick={() => setShowConfirm(true)}
                  style={{ display: "flex", alignItems: "center", gap: 5, padding: "5px 10px", borderRadius: 8, background: "rgba(77,255,160,0.06)", border: "1px solid rgba(77,255,160,0.18)", cursor: "pointer", transition: "all 0.2s" }}
                  onMouseEnter={e => { e.currentTarget.style.background = "rgba(77,255,160,0.14)"; }}
                  onMouseLeave={e => { e.currentTarget.style.background = "rgba(77,255,160,0.06)"; }}>
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#4DFFA0" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-4" />
                  </svg>
                  <span style={{ fontSize: 9.5, color: "#4DFFA0", fontFamily: "'DM Mono',monospace", fontWeight: 500 }}>NEW</span>
                </button>
              )}
            </div>
            <div style={{ fontSize: 9.5, color: "#334155", fontFamily: "'DM Mono',monospace", letterSpacing: "0.1em", marginTop: 2 }}>CLAUDE-AS-JUDGE · STUDY MODE</div>
          </div>

          {/* Root Anchor */}
          {rootQuery && (
            <div style={{ padding: "10px 12px", borderRadius: 10, background: "rgba(77,255,160,0.05)", border: "1px solid rgba(77,255,160,0.15)", marginBottom: 14 }}>
              <div style={{ fontSize: 9.5, color: "#4DFFA0", fontFamily: "'DM Mono',monospace", letterSpacing: "0.1em", marginBottom: 4 }}>⚓ ROOT TOPIC</div>
              <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.5 }}>{rootQuery}</div>
            </div>
          )}

          {/* Judge Verdict */}
          {turnIndex >= JUDGE_STARTS_AT_TURN && <JudgeVerdict verdict={lastVerdict} />}

          {/* Drift counter */}
          {driftCount > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 10px", borderRadius: 8, background: "rgba(255,77,109,0.07)", border: "1px solid rgba(255,77,109,0.15)", marginBottom: 14 }}>
              <span style={{ fontSize: 10, color: "#FF4D6D", fontFamily: "'DM Mono',monospace" }}>BRIDGES BUILT</span>
              <span style={{ fontSize: 16, fontWeight: 800, color: "#FF4D6D", fontFamily: "'Syne',sans-serif" }}>{driftCount}</span>
            </div>
          )}

          {/* Concept map */}
          {knowledge?.chunks?.length > 0 && (
            <>
              <div style={{ fontSize: 9.5, color: "#334155", fontFamily: "'DM Mono',monospace", letterSpacing: "0.1em", marginBottom: 8, textTransform: "uppercase" }}>Concept Map</div>
              {knowledge.chunks.map((chunk, i) => (
                <div key={chunk} style={{ padding: "7px 9px", borderRadius: 8, marginBottom: 3, background: i === knowledge.current_chunk_index ? "rgba(77,255,160,0.06)" : "transparent", border: i === knowledge.current_chunk_index ? "1px solid rgba(77,255,160,0.2)" : "1px solid transparent" }}>
                  <ScoreBar label={chunk} score={knowledge.chunk_scores?.[chunk] ?? 0} />
                </div>
              ))}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6, marginBottom: 14, padding: "6px 10px", borderRadius: 8, background: "rgba(15,23,42,0.8)" }}>
                <span style={{ fontSize: 9.5, color: "#334155", fontFamily: "'DM Mono',monospace" }}>CONFIDENCE</span>
                <span style={{ fontSize: 20, fontWeight: 800, fontFamily: "'Syne',sans-serif", color: knowledge.overall_confidence >= 70 ? "#4DFFA0" : knowledge.overall_confidence >= 40 ? "#FFD166" : "#FF4D6D" }}>
                  {knowledge.overall_confidence}%
                </span>
              </div>
              {knowledge.action_taken && (
                <div style={{ textAlign: "center", marginBottom: 10 }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 10px", borderRadius: 12, background: `${actionColors[knowledge.action_taken] || "#64748b"}15`, border: `1px solid ${actionColors[knowledge.action_taken] || "#64748b"}35` }}>
                    <span style={{ width: 5, height: 5, borderRadius: "50%", background: actionColors[knowledge.action_taken] || "#64748b", boxShadow: `0 0 5px ${actionColors[knowledge.action_taken] || "#64748b"}` }} />
                    <span style={{ fontSize: 9.5, color: actionColors[knowledge.action_taken] || "#64748b", fontFamily: "'DM Mono',monospace", fontWeight: 500 }}>
                      {knowledge.action_taken.replace(/_/g, " ").toUpperCase()}
                    </span>
                  </span>
                </div>
              )}
            </>
          )}

          {/* Summary Store */}
          {summaries.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: 9.5, color: "#334155", fontFamily: "'DM Mono',monospace", letterSpacing: "0.1em", marginBottom: 8, textTransform: "uppercase" }}>📝 Session Log</div>
              <div style={{ maxHeight: 180, overflowY: "auto" }}>
                {summaries.map((s, i) => (
                  <div key={i} style={{ padding: "5px 8px", borderRadius: 6, marginBottom: 3, background: s.isDrift ? "rgba(255,77,109,0.06)" : "rgba(15,23,42,0.5)", border: `1px solid ${s.isDrift ? "rgba(255,77,109,0.12)" : "rgba(255,255,255,0.04)"}` }}>
                    <div style={{ fontSize: 9, color: s.isDrift ? "#FF4D6D" : "#334155", fontFamily: "'DM Mono',monospace", marginBottom: 2 }}>
                      {s.isDrift ? "🌉" : "📖"} TURN {s.turn + 1}
                    </div>
                    <div style={{ fontSize: 10.5, color: "#475569", lineHeight: 1.4 }}>{s.text}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── MAIN CHAT ──────────────────────────────────────────────────── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}>

          <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 15% 15%, rgba(77,255,160,0.025) 0%, transparent 55%), radial-gradient(ellipse at 85% 85%, rgba(14,165,233,0.025) 0%, transparent 55%)", pointerEvents: "none" }} />

          {/* Status bar */}
          {statusMsg && (
            <div style={{ padding: "6px 20px", background: "rgba(77,255,160,0.04)", borderBottom: "1px solid rgba(77,255,160,0.06)", fontSize: 11, color: "#4DFFA0", fontFamily: "'DM Mono',monospace", letterSpacing: "0.05em", zIndex: 2, position: "relative" }}>
              {statusMsg}
            </div>
          )}

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "28px 44px", position: "relative", zIndex: 1 }}>
            {messages.length === 0 ? (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", textAlign: "center", animation: "fadeIn 0.5s ease" }}>
                <div style={{ fontSize: 52, marginBottom: 18 }}>🦉</div>
                <h1 style={{ fontFamily: "'Syne',sans-serif", fontSize: 34, fontWeight: 800, background: "linear-gradient(135deg,#4DFFA0,#0EA5E9)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 12, letterSpacing: "-0.5px" }}>
                  Context-Aware Study Mode
                </h1>
                <p style={{ color: "#475569", fontSize: 14.5, maxWidth: 420, lineHeight: 1.75, marginBottom: 10 }}>
                  A Socratic tutor that detects when teaching drifts off-topic — using <strong style={{ color: "#94a3b8" }}>Claude as the judge</strong>, not keyword matching.
                </p>
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 30, padding: "6px 14px", borderRadius: 20, background: "rgba(255,77,109,0.08)", border: "1px solid rgba(255,77,109,0.2)" }}>
                  <span style={{ fontSize: 11, color: "#FF4D6D", fontFamily: "'DM Mono',monospace" }}>🌉 Try asking an off-topic question mid-lesson to see the bridge</span>
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 9, justifyContent: "center", maxWidth: 500 }}>
                  {SUGGESTED.map(t => (
                    <button key={t} onClick={() => sendMessage(`I want to learn about: ${t}`)}
                      style={{ background: "rgba(77,255,160,0.06)", border: "1px solid rgba(77,255,160,0.18)", borderRadius: 20, padding: "7px 16px", color: "#4DFFA0", fontSize: 12, fontFamily: "'DM Mono',monospace", cursor: "pointer", transition: "all 0.2s" }}
                      onMouseEnter={e => { e.target.style.background = "rgba(77,255,160,0.14)"; e.target.style.transform = "translateY(-2px)"; }}
                      onMouseLeave={e => { e.target.style.background = "rgba(77,255,160,0.06)"; e.target.style.transform = "translateY(0)"; }}
                    >{t}</button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {messages.map((msg, i) => <MessageBubble key={i} msg={msg} isNew={i === newMsgIdx} />)}
                {loading && (
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
                    <div style={{ width: 34, height: 34, borderRadius: "50%", background: "linear-gradient(135deg,#4DFFA0,#0EA5E9)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, boxShadow: "0 0 14px #4DFFA050" }}>🦉</div>
                    <div style={{ display: "flex", gap: 5 }}>
                      {[0, 1, 2].map(j => <div key={j} style={{ width: 7, height: 7, borderRadius: "50%", background: "#4DFFA0", animation: `pulse 1.2s ease ${j * 0.2}s infinite` }} />)}
                    </div>
                    <span style={{ fontSize: 11, color: "#334155", fontFamily: "'DM Mono',monospace" }}>{statusMsg || "thinking..."}</span>
                  </div>
                )}
                <div ref={bottomRef} />
              </>
            )}
          </div>

          {/* Input */}
          <div style={{ padding: "14px 44px 22px", borderTop: "1px solid rgba(77,255,160,0.05)", background: "rgba(2,8,23,0.85)", backdropFilter: "blur(12px)", position: "relative", zIndex: 1 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "flex-end", background: "rgba(10,18,36,0.95)", border: "1px solid rgba(77,255,160,0.12)", borderRadius: 14, padding: "11px 14px" }}>
              <textarea ref={inputRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={handleKey}
                placeholder={messages.length === 0 ? "Name a topic to learn, e.g. 'quantum mechanics'..." : "Answer the question, ask a doubt, or try going off-topic..."}
                rows={1}
                style={{ flex: 1, background: "transparent", border: "none", color: "#e2e8f0", fontSize: 14, fontFamily: "'Lora',serif", lineHeight: 1.6, resize: "none", maxHeight: 120, overflowY: "auto" }}
                onInput={e => { e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; }}
              />
              <button onClick={() => sendMessage(input)} disabled={loading || !input.trim()}
                style={{ width: 36, height: 36, borderRadius: 9, background: (loading || !input.trim()) ? "rgba(77,255,160,0.08)" : "linear-gradient(135deg,#4DFFA0,#0EA5E9)", border: "none", cursor: (loading || !input.trim()) ? "not-allowed" : "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, transition: "all 0.2s", boxShadow: (!loading && input.trim()) ? "0 0 14px rgba(77,255,160,0.35)" : "none" }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={(loading || !input.trim()) ? "#334155" : "#020817"} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <div style={{ textAlign: "center", marginTop: 6, fontSize: 10, color: "#1e293b", fontFamily: "'DM Mono',monospace" }}>
              Turn {turnIndex} · {summaries.length} summaries stored · Enter to send
            </div>
          </div>
        </div>
      </div>

      {/* ── NEW SESSION MODAL ───────────────────────────────────────────── */}
      {showConfirm && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(2,8,23,0.85)", backdropFilter: "blur(8px)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100, animation: "fadeIn 0.2s ease" }}>
          <div style={{ background: "rgba(10,18,36,0.98)", border: "1px solid rgba(77,255,160,0.18)", borderRadius: 18, padding: "32px 36px", maxWidth: 360, width: "90%", boxShadow: "0 24px 80px rgba(0,0,0,0.6)", textAlign: "center" }}>
            <div style={{ fontSize: 36, marginBottom: 14 }}>🔄</div>
            <div style={{ fontFamily: "'Syne',sans-serif", fontSize: 20, fontWeight: 800, color: "#e2e8f0", marginBottom: 10 }}>Start New Session?</div>
            <div style={{ fontSize: 13, color: "#475569", fontFamily: "'Lora',serif", lineHeight: 1.65, marginBottom: 28 }}>
              This will clear the conversation, summary store, and all progress on <span style={{ color: "#94a3b8", fontStyle: "italic" }}>"{rootQuery}"</span>.
            </div>
            <div style={{ display: "flex", gap: 10 }}>
              <button onClick={() => setShowConfirm(false)}
                style={{ flex: 1, padding: "10px 0", borderRadius: 10, background: "rgba(30,41,59,0.6)", border: "1px solid rgba(255,255,255,0.06)", color: "#64748b", fontSize: 13, fontFamily: "'DM Mono',monospace", cursor: "pointer" }}
                onMouseEnter={e => { e.currentTarget.style.color = "#94a3b8"; }}
                onMouseLeave={e => { e.currentTarget.style.color = "#64748b"; }}
              >Cancel</button>
              <button onClick={newSession}
                style={{ flex: 1, padding: "10px 0", borderRadius: 10, background: "linear-gradient(135deg,#4DFFA0,#0EA5E9)", border: "none", color: "#020817", fontSize: 13, fontFamily: "'DM Mono',monospace", fontWeight: 700, cursor: "pointer", boxShadow: "0 0 20px rgba(77,255,160,0.3)" }}
                onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-1px)"; }}
                onMouseLeave={e => { e.currentTarget.style.transform = "translateY(0)"; }}
              >New Session</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}