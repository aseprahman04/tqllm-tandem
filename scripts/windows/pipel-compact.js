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
