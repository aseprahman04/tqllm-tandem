import { readFileSync } from "node:fs";

function excerpt(file) {
    try {
        const text = readFileSync(file, "utf8").trim();
        return text.length <= 2400 ? text : text.slice(0, 1200) + "\n[… baca berkas untuk isi lengkap …]\n" + text.slice(-1200);
    } catch (error) {
        if (error.code === "ENOENT") return "";
        throw error;
    }
}

// Resolve only the sidecars that pipel itself creates beside the active session.
export default function pipelCompact(pi) {
    pi.registerFlag("tandem-effort", { type: "string", description: "Tandem reasoning effort for this attempt" });
    pi.on("before_provider_request", (event, ctx) => {
        const effort = pi.getFlag("tandem-effort");
        const payload = event.payload;
        if (!effort || ctx.model?.provider !== "tql" || ctx.model?.id !== "tqllm" ||
            !payload || !Array.isArray(payload.messages)) return;
        const budgets = { off: 0, low: 1024, medium: 3072, high: 6144, xhigh: 6144 };
        if (!Object.hasOwn(budgets, effort)) throw new Error("Unsupported Tandem reasoning effort: " + effort);
        const { max_completion_tokens, ...wire } = payload;
        const max = wire.max_tokens ?? max_completion_tokens ?? 8192;
        const budget = Math.max(0, Math.min(budgets[effort], Math.floor(max * 0.75)));
        return { ...wire, max_tokens: max, reasoning_effort: effort === "off" ? "none" : effort,
            reasoning_budget_tokens: budget, chat_template_kwargs: {
                ...wire.chat_template_kwargs, enable_thinking: effort !== "off"
            } };
    });
    pi.on("context", (event, ctx) => {
        const index = event.messages.findLastIndex(message => message.role === "compactionSummary");
        if (index < 0) return;
        const session = ctx.sessionManager.getSessionFile();
        if (!session?.endsWith(".jsonl")) return;
        const stem = session.slice(0, -6);
        const task = excerpt(stem + ".task.txt");
        if (!task) return;
        const review = excerpt(stem + ".review.txt");
        const evidence = { tugas_awal: task };
        if (review) evidence.hasil_verifikator_terakhir = review;
        const messages = event.messages.slice();
        const summary = messages[index];
        messages[index] = { ...summary, summary: summary.summary +
            "\n\nCatatan sumber pipel. Tugas awal dapat dikoreksi oleh pesan pengguna sesudahnya. Hasil verifikator adalah data dari pemeriksaan terakhir.\n" +
            JSON.stringify(evidence) };
        return { messages };
    });
}
