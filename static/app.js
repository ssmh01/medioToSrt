const state = {
    currentJobId: null,
    pollTimer: null,
    startedAt: null,
    logs: [],
    languageDefaults: {},
};

const nodes = {
    form: document.getElementById("jobForm"),
    audioInput: document.getElementById("audioInput"),
    scriptFileInput: document.getElementById("scriptFileInput"),
    scriptText: document.getElementById("scriptText"),
    textCount: document.getElementById("textCount"),
    languageSelect: document.getElementById("languageSelect"),
    profileSelect: document.getElementById("profileSelect"),
    minDuration: document.getElementById("minDuration"),
    maxDuration: document.getElementById("maxDuration"),
    maxChars: document.getElementById("maxChars"),
    generateVtt: document.getElementById("generateVtt"),
    preservePunctuation: document.getElementById("preservePunctuation"),
    submitButton: document.getElementById("submitButton"),
    statusText: document.getElementById("statusText"),
    previewBody: document.getElementById("previewBody"),
    rowCount: document.getElementById("rowCount"),
    logList: document.getElementById("logList"),
    clearLogs: document.getElementById("clearLogs"),
    qualityGrid: document.getElementById("qualityGrid"),
    qualityBadge: document.getElementById("qualityBadge"),
    downloadList: document.getElementById("downloadList"),
    refreshButton: document.getElementById("refreshButton"),
    waveform: document.getElementById("waveform"),
    audioDuration: document.getElementById("audioDuration"),
};

init();

async function init() {
    buildWaveform();
    setupUploadCard("audioInput", "audioCard", "audioEmpty", "audioChosen", "audioName", "audioMeta");
    setupUploadCard("scriptFileInput", "scriptCard", "scriptEmpty", "scriptChosen", "scriptName", "scriptMeta");
    setupSteppers();
    await loadOptions();
    bindEvents();
    renderStage("idle");
}

function bindEvents() {
    nodes.form.addEventListener("submit", submitJob);
    nodes.languageSelect.addEventListener("change", applyLanguageDefaults);
    nodes.scriptText.addEventListener("input", () => {
        nodes.textCount.textContent = String(nodes.scriptText.value.trim().length);
    });
    nodes.clearLogs.addEventListener("click", () => {
        state.logs = [];
        renderLogs([]);
    });
    nodes.refreshButton.addEventListener("click", () => {
        if (state.currentJobId) {
            pollJob(state.currentJobId);
        }
    });
    document.querySelectorAll("[data-clear]").forEach((button) => {
        button.addEventListener("click", (event) => {
            event.stopPropagation();
            const input = document.getElementById(button.dataset.clear);
            input.value = "";
            input.dispatchEvent(new Event("change"));
        });
    });
}

async function loadOptions() {
    const response = await fetch("/api/options");
    const data = await response.json();
    const languages = data.languages.filter((value) => value !== "auto");
    state.languageDefaults = data.language_defaults || {};
    nodes.languageSelect.innerHTML = '<option value="" selected disabled>请选择语言</option>' + languages
        .map((value) => `<option value="${escapeAttr(value)}">${languageLabel(value)}</option>`)
        .join("");
    nodes.profileSelect.innerHTML = data.profiles
        .map((profile) => `<option value="${escapeAttr(profile.key)}">${escapeHtml(profile.label)}</option>`)
        .join("");
    nodes.languageSelect.value = "";
    nodes.profileSelect.value = data.defaults.subtitle_profile;
    nodes.minDuration.value = data.defaults.min_duration;
    nodes.maxDuration.value = data.defaults.max_duration;
    nodes.maxChars.value = data.defaults.max_chars_per_line;
    nodes.generateVtt.checked = data.defaults.generate_vtt;
    nodes.preservePunctuation.checked = data.defaults.preserve_punctuation;
}

function applyLanguageDefaults() {
    const defaults = state.languageDefaults[nodes.languageSelect.value];
    if (!defaults) {
        return;
    }
    nodes.minDuration.value = defaults.min_duration;
    nodes.maxDuration.value = defaults.max_duration;
    nodes.maxChars.value = defaults.max_chars_per_line;
}

function setupUploadCard(inputId, cardId, emptyId, chosenId, nameId, metaId) {
    const input = document.getElementById(inputId);
    const card = document.getElementById(cardId);
    const empty = document.getElementById(emptyId);
    const chosen = document.getElementById(chosenId);
    const name = document.getElementById(nameId);
    const meta = document.getElementById(metaId);

    input.addEventListener("change", () => {
        const file = input.files[0];
        if (!file) {
            empty.classList.remove("hidden");
            chosen.classList.add("hidden");
            return;
        }
        name.textContent = file.name;
        meta.textContent = formatBytes(file.size);
        empty.classList.add("hidden");
        chosen.classList.remove("hidden");
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        card.addEventListener(eventName, (event) => {
            event.preventDefault();
            card.classList.add("dragging");
        });
    });
    ["dragleave", "drop"].forEach((eventName) => {
        card.addEventListener(eventName, (event) => {
            event.preventDefault();
            card.classList.remove("dragging");
        });
    });
    card.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files[0];
        if (!file) {
            return;
        }
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        input.dispatchEvent(new Event("change"));
    });
}

function setupSteppers() {
    document.querySelectorAll("[data-step][data-target]").forEach((button) => {
        button.addEventListener("click", () => {
            const input = document.getElementById(button.dataset.target);
            const step = Number(button.dataset.step);
            const current = Number(input.value || 0);
            const next = current + step;
            const min = Number(input.min || 0);
            const decimals = Math.abs(step) < 1 ? 1 : 0;
            input.value = String(Math.max(min, next).toFixed(decimals));
        });
    });
}

async function submitJob(event) {
    event.preventDefault();
    if (!nodes.audioInput.files.length) {
        showError("请先上传音频文件");
        return;
    }
    if (!nodes.scriptFileInput.files.length && !nodes.scriptText.value.trim()) {
        showError("请上传 TXT 文案或粘贴原始文案");
        return;
    }
    if (!nodes.languageSelect.value) {
        showError("请先选择语言");
        nodes.languageSelect.focus();
        return;
    }

    clearPolling();
    state.logs = ["正在提交任务"];
    state.startedAt = Date.now();
    renderLogs(state.logs);
    renderStage("queued");
    renderDownloads([]);
    renderQuality(null);
    nodes.submitButton.disabled = true;
    nodes.submitButton.innerHTML = '<span class="play-icon">▶</span> 生成中';

    const formData = new FormData();
    formData.append("audio_file", nodes.audioInput.files[0]);
    if (nodes.scriptFileInput.files.length) {
        formData.append("script_file", nodes.scriptFileInput.files[0]);
    }
    formData.append("script_text", nodes.scriptText.value);
    formData.append("language", nodes.languageSelect.value);
    formData.append("subtitle_profile", nodes.profileSelect.value);
    formData.append("min_duration", nodes.minDuration.value);
    formData.append("max_duration", nodes.maxDuration.value);
    formData.append("max_chars_per_line", nodes.maxChars.value);
    formData.append("generate_vtt", nodes.generateVtt.checked ? "true" : "false");
    formData.append("preserve_punctuation", nodes.preservePunctuation.checked ? "true" : "false");

    try {
        const response = await fetch("/api/jobs", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "任务提交失败");
        }
        state.currentJobId = payload.job_id;
        pollJob(payload.job_id);
        state.pollTimer = window.setInterval(() => pollJob(payload.job_id), 1000);
    } catch (error) {
        showError(error.message);
        renderStage("failed");
        resetSubmitButton();
    }
}

async function pollJob(jobId) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const payload = await response.json();
    if (!response.ok) {
        showError(payload.detail || "读取任务状态失败");
        clearPolling();
        resetSubmitButton();
        return;
    }

    state.logs = payload.logs || [];
    renderLogs(state.logs);
    renderStage(payload.stage || payload.status);
    renderPreview(payload.preview_rows || []);
    renderQuality(payload.quality_report);
    renderDownloads(payload.downloads || [], payload.quality_report);

    if (payload.quality_report?.audio_duration) {
        nodes.audioDuration.textContent = formatDuration(payload.quality_report.audio_duration);
    }

    if (payload.status === "succeeded") {
        clearPolling();
        nodes.statusText.textContent = "生成完成";
        resetSubmitButton();
    }
    if (payload.status === "failed") {
        clearPolling();
        showError(payload.error || "任务失败");
        resetSubmitButton();
    }
}

function renderStage(stage) {
    const steps = [...document.querySelectorAll(".step")];
    steps.forEach((step) => step.classList.remove("active", "complete", "error"));
    const order = ["upload", "align", "validate", "export"];
    let activeIndex = -1;

    if (stage === "idle") {
        nodes.statusText.textContent = "等待上传";
        return;
    }
    if (stage === "queued") {
        activeIndex = 0;
        nodes.statusText.textContent = "已提交任务";
    } else if (stage === "aligning" || stage === "running") {
        activeIndex = 1;
        nodes.statusText.textContent = "正在对齐";
    } else if (stage === "succeeded") {
        activeIndex = 3;
        nodes.statusText.textContent = "生成完成";
    } else if (stage === "failed") {
        activeIndex = 1;
        nodes.statusText.textContent = "生成失败";
    }

    steps.forEach((step) => {
        const index = order.indexOf(step.dataset.stepName);
        if (stage === "succeeded" || index < activeIndex) {
            step.classList.add("complete");
            if (step.dataset.stepName === "upload") {
                step.querySelector(".step-dot").textContent = "✓";
            }
        } else if (index === activeIndex) {
            step.classList.add(stage === "failed" ? "error" : "active");
        }
    });
}

function renderLogs(logs) {
    if (!logs.length) {
        nodes.logList.innerHTML = '<li class="muted">暂无日志</li>';
        return;
    }
    nodes.logList.innerHTML = logs
        .map((line) => `<li>${escapeHtml(line)}</li>`)
        .join("");
}

function renderPreview(rows) {
    nodes.rowCount.textContent = String(rows.length);
    if (!rows.length) {
        nodes.previewBody.innerHTML = '<tr class="empty-row"><td colspan="6">生成后会在这里显示字幕预览</td></tr>';
        return;
    }
    nodes.previewBody.innerHTML = rows.map((row) => {
        const warning = row[5] || "";
        return `
            <tr class="${warning ? "warning-row" : ""}">
                <td>${escapeHtml(row[0])}</td>
                <td>${escapeHtml(row[1])}</td>
                <td>${escapeHtml(row[2])}</td>
                <td>${escapeHtml(row[3])}</td>
                <td>${escapeHtml(row[4])}</td>
                <td>${warning ? `<span class="warning-chip">${escapeHtml(warning)}</span>` : "-"}</td>
            </tr>
        `;
    }).join("");
}

function renderQuality(report) {
    const values = report ? qualityValues(report) : [
        ["总字幕条数", "0"],
        ["警告条数", "0"],
        ["错误条数", "0"],
        ["语音覆盖率", "--"],
        ["平均每条时长", "--"],
        ["最长时长", "--"],
        ["时间轴状态", "--"],
        ["时间轴置信度", "--"],
        ["低置信段", "0"],
    ];
    nodes.qualityGrid.innerHTML = values
        .map(([label, value]) => `<div><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`)
        .join("");

    nodes.qualityBadge.className = "pass-badge";
    if (!report) {
        nodes.qualityBadge.textContent = "等待报告";
        return;
    }
    const warnings = report.warnings || [];
    const score = Number(report.quality_score || 0);
    if (!warnings.length && score >= 90) {
        nodes.qualityBadge.textContent = "通过";
        nodes.qualityBadge.classList.add("pass");
    } else {
        nodes.qualityBadge.textContent = "需检查";
        nodes.qualityBadge.classList.add("warn");
    }
}

function qualityValues(report) {
    const warningCount = (report.warnings || []).length;
    const errorCount = Number(report.too_long_count || 0)
        + Number(report.too_short_count || 0)
        + Number(report.overlap_count || 0);
    const coverage = `${Math.max(0, (1 - Number(report.unaligned_text_ratio || 0)) * 100).toFixed(2)}%`;
    return [
        ["总字幕条数", String(report.subtitle_count || 0)],
        ["警告条数", String(warningCount)],
        ["错误条数", String(errorCount)],
        ["语音覆盖率", coverage],
        ["平均每条时长", `${report.avg_subtitle_duration || 0} 秒`],
        ["最长时长", `${report.max_subtitle_duration || 0} 秒`],
        ["时间轴状态", timelineStatusLabel(report.timeline_status || "ok")],
        ["时间轴置信度", `${report.timeline_confidence_score ?? "--"}`],
        ["低置信段", String((report.low_confidence_ranges || []).length)],
    ];
}

function timelineStatusLabel(status) {
    if (status === "needs_review") return "需检查";
    if (status === "repaired") return "已修复";
    return "正常";
}

function renderDownloads(downloads, qualityReport = null) {
    if (!downloads.length) {
        nodes.downloadList.innerHTML = '<div class="empty-download">生成完成后显示下载文件</div>';
        return;
    }
    const needsTimelineReview = qualityReport?.timeline_status === "needs_review";
    nodes.downloadList.innerHTML = downloads.map((item) => {
        const extension = item.label.split(".").pop().toUpperCase();
        const meta = needsTimelineReview && item.kind === "srt"
            ? "需检查时间轴"
            : "点击右侧按钮下载";
        return `
            <div class="download-item ${needsTimelineReview && item.kind === "srt" ? "needs-review" : ""}">
                <div class="download-icon ${escapeAttr(item.kind)}">${escapeHtml(extension)}</div>
                <div>
                    <div class="download-name">${escapeHtml(item.label)}</div>
                    <div class="download-meta">${escapeHtml(meta)}</div>
                </div>
                <a class="download-link" href="${escapeAttr(item.url)}" download title="下载">⇩</a>
            </div>
        `;
    }).join("");
}

function showError(message) {
    state.logs = [...state.logs, message];
    renderLogs(state.logs);
    nodes.statusText.textContent = message;
}

function resetSubmitButton() {
    nodes.submitButton.disabled = false;
    nodes.submitButton.innerHTML = '<span class="play-icon">▶</span> 开始生成';
}

function clearPolling() {
    if (state.pollTimer) {
        window.clearInterval(state.pollTimer);
        state.pollTimer = null;
    }
}

function buildWaveform() {
    const bars = Array.from({ length: 160 }, (_, index) => {
        const height = 16 + Math.round(Math.abs(Math.sin(index * 0.55)) * 28) + (index % 9);
        return `<span style="height:${height}px"></span>`;
    }).join("");
    nodes.waveform.innerHTML = bars;
}

function languageLabel(value) {
    return {
        zh: "中文（简体）",
        "zh-TW": "中文（繁体）",
        ja: "日语",
        en: "英语",
        ko: "韩语",
    }[value] || value;
}

function formatBytes(bytes) {
    if (!bytes) {
        return "0 B";
    }
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    return `${(bytes / Math.pow(1024, index)).toFixed(index ? 2 : 0)} ${units[index]}`;
}

function formatDuration(seconds) {
    const total = Math.max(0, Math.round(seconds));
    const minutes = Math.floor(total / 60);
    const rest = total % 60;
    return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
    return escapeHtml(value);
}
