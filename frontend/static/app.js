const state = {
  catalog: null,
  backendId: "fromw1",
  motionId: null,
  versionId: null,
};

const elements = {
  catalogStatus: document.querySelector("#catalog-status"),
  refreshButton: document.querySelector("#refresh-button"),
  backendControl: document.querySelector("#backend-control"),
  motionSelect: document.querySelector("#motion-select"),
  versionSelect: document.querySelector("#version-select"),
  availabilityDot: document.querySelector("#availability-dot"),
  availabilityText: document.querySelector("#availability-text"),
  sourceMedia: document.querySelector("#source-media"),
  referenceMedia: document.querySelector("#reference-media"),
  executionMedia: document.querySelector("#execution-media"),
  sourceState: document.querySelector("#source-state"),
  referenceState: document.querySelector("#reference-state"),
  executionState: document.querySelector("#execution-state"),
  sourcePath: document.querySelector("#source-path"),
  referencePath: document.querySelector("#reference-path"),
  executionPath: document.querySelector("#execution-path"),
  visualReview: document.querySelector("#visual-review"),
  reviewTime: document.querySelector("#review-time"),
  reviewSource: document.querySelector("#review-source"),
  editPlan: document.querySelector("#edit-plan"),
  planTime: document.querySelector("#plan-time"),
  planSource: document.querySelector("#plan-source"),
  feedbackForm: document.querySelector("#feedback-form"),
  feedbackInput: document.querySelector("#feedback-input"),
  feedbackStatus: document.querySelector("#feedback-status"),
  submitButton: document.querySelector("#submit-button"),
  characterCount: document.querySelector("#character-count"),
  feedbackHistory: document.querySelector("#feedback-history"),
  historyCount: document.querySelector("#history-count"),
};

function getBackend() {
  return state.catalog?.backends.find((backend) => backend.id === state.backendId) ?? null;
}

function getVersionsForMotion(motionId) {
  const backend = getBackend();
  if (!backend) return [];
  return backend.versions.filter((version) =>
    version.motions.some((motion) => motion.id === motionId)
  );
}

function getMotionOptions() {
  const backend = getBackend();
  if (!backend) return [];
  const motions = new Map();
  backend.versions.forEach((version) => {
    version.motions.forEach((motion) => motions.set(motion.id, motion.label));
  });
  return [...motions.entries()]
    .map(([id, label]) => ({ id, label }))
    .sort((a, b) => a.id.localeCompare(b.id));
}

function getSelection() {
  const backend = getBackend();
  const version = backend?.versions.find((item) => item.id === state.versionId);
  const motion = version?.motions.find((item) => item.id === state.motionId);
  return { backend, version, motion };
}

function buildBackendControl() {
  elements.backendControl.replaceChildren();
  state.catalog.backends.forEach((backend) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "segment-button";
    button.textContent = backend.label;
    button.dataset.backendId = backend.id;
    button.setAttribute("aria-pressed", String(backend.id === state.backendId));
    button.addEventListener("click", () => {
      state.backendId = backend.id;
      state.motionId = null;
      state.versionId = null;
      renderSelectors();
    });
    elements.backendControl.append(button);
  });
}

function fillSelect(select, options, selectedId) {
  select.replaceChildren();
  options.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label;
    option.selected = item.id === selectedId;
    select.append(option);
  });
  select.disabled = options.length === 0;
}

function renderSelectors() {
  buildBackendControl();
  const motions = getMotionOptions();
  if (!motions.some((motion) => motion.id === state.motionId)) {
    state.motionId = motions[0]?.id ?? null;
  }
  fillSelect(
    elements.motionSelect,
    motions.map((motion) => ({
      id: motion.id,
      label: `${motion.label}  (${motion.id})`,
    })),
    state.motionId
  );

  const versions = getVersionsForMotion(state.motionId);
  if (!versions.some((version) => version.id === state.versionId)) {
    state.versionId =
      versions.find((version) => version.id === "canonical_v2")?.id ??
      versions[0]?.id ??
      null;
  }
  fillSelect(elements.versionSelect, versions, state.versionId);
  renderSelection();
}

function renderMedia(container, stateLabel, pathLabel, asset) {
  container.replaceChildren();
  if (!asset?.available) {
    const missing = document.createElement("div");
    missing.className = "missing-media";
    missing.textContent = "尚未生成该素材";
    container.append(missing);
    stateLabel.textContent = "缺失";
    stateLabel.className = "asset-state missing";
    pathLabel.textContent = "";
    pathLabel.title = "";
    return;
  }

  let media;
  if (asset.kind === "video") {
    media = document.createElement("video");
    media.controls = true;
    media.loop = true;
    media.muted = true;
    media.playsInline = true;
    media.preload = "metadata";
  } else {
    media = document.createElement("img");
    media.alt = asset.label;
    media.loading = "eager";
  }
  media.src = asset.url;
  container.append(media);
  stateLabel.textContent = "可用";
  stateLabel.className = "asset-state available";
  pathLabel.textContent = asset.path;
  pathLabel.title = asset.path;
}

function renderVisualReview(review) {
  if (!review?.text) {
    elements.visualReview.textContent = "暂无视觉评审";
    elements.visualReview.className = "review-content empty-copy";
    elements.reviewTime.textContent = "";
    elements.reviewSource.textContent = "";
    return;
  }
  elements.visualReview.textContent = review.text;
  elements.visualReview.className = "review-content";
  elements.reviewTime.textContent = review.updated_at ?? "";
  elements.reviewSource.textContent = review.source ?? "";
  elements.reviewSource.title = review.source ?? "";
}

function appendPlanRow(container, operation, reason) {
  const row = document.createElement("div");
  row.className = "plan-row";
  const operationElement = document.createElement("div");
  operationElement.className = "plan-operation";
  operationElement.textContent = operation;
  const reasonElement = document.createElement("div");
  reasonElement.className = "plan-reason";
  reasonElement.textContent = reason || "未提供原因";
  row.append(operationElement, reasonElement);
  container.append(row);
}

function renderEditPlan(planRecord) {
  elements.editPlan.replaceChildren();
  if (!planRecord?.data) {
    elements.editPlan.textContent = "暂无修改建议";
    elements.editPlan.className = "plan-content empty-copy";
    elements.planTime.textContent = "";
    elements.planSource.textContent = "";
    return;
  }

  elements.editPlan.className = "plan-content";
  const plan = planRecord.data;
  const recommended =
    plan.summary?.recommended_next_step ||
    plan.expected_effect ||
    plan.summary?.user_feedback_interpretation;
  if (recommended) {
    const summary = document.createElement("p");
    summary.className = "plan-summary";
    summary.textContent = recommended;
    elements.editPlan.append(summary);
  }

  const edits = Array.isArray(plan.edits) ? plan.edits : [];
  edits.forEach((edit) => {
    const operation = [edit.type, edit.target].filter(Boolean).join(" · ");
    appendPlanRow(elements.editPlan, operation || "edit", edit.reason);
  });
  if (edits.length === 0 && !recommended) {
    const raw = document.createElement("pre");
    raw.textContent = JSON.stringify(plan, null, 2);
    elements.editPlan.append(raw);
  }

  elements.planTime.textContent = planRecord.updated_at ?? "";
  elements.planSource.textContent = planRecord.source ?? "";
  elements.planSource.title = planRecord.source ?? "";
}

function renderSelection() {
  const { backend, version, motion } = getSelection();
  if (!backend || !version || !motion) {
    elements.availabilityDot.className = "status-dot";
    elements.availabilityText.textContent = "没有可显示的数据";
    return;
  }

  renderMedia(elements.sourceMedia, elements.sourceState, elements.sourcePath, motion.source);
  renderMedia(
    elements.referenceMedia,
    elements.referenceState,
    elements.referencePath,
    motion.reference
  );
  renderMedia(
    elements.executionMedia,
    elements.executionState,
    elements.executionPath,
    motion.execution
  );
  renderVisualReview(motion.visual_review);
  renderEditPlan(motion.edit_plan);

  const availableCount = [motion.source, motion.reference, motion.execution].filter(
    (asset) => asset?.available
  ).length;
  elements.availabilityDot.className =
    availableCount === 3 ? "status-dot complete" : "status-dot partial";
  elements.availabilityText.textContent = `${backend.label} / ${version.label} / ${availableCount} 项素材可用`;
  elements.feedbackStatus.textContent = "";
  loadFeedbackHistory();
}

function renderFeedbackHistory(items) {
  elements.feedbackHistory.replaceChildren();
  elements.historyCount.textContent = `${items.length} 条`;
  if (items.length === 0) {
    elements.feedbackHistory.textContent = "暂无用户反馈";
    elements.feedbackHistory.className = "feedback-history empty-copy";
    return;
  }

  elements.feedbackHistory.className = "feedback-history";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "feedback-item";
    const meta = document.createElement("div");
    meta.className = "feedback-meta";
    meta.textContent = item.created_at ?? "";
    const comment = document.createElement("div");
    comment.className = "feedback-comment";
    comment.textContent = item.comment ?? "";
    row.append(meta, comment);
    elements.feedbackHistory.append(row);
  });
}

async function loadFeedbackHistory() {
  const { backend, version, motion } = getSelection();
  if (!backend || !version || !motion) {
    renderFeedbackHistory([]);
    return;
  }
  const params = new URLSearchParams({
    backend: backend.id,
    version: version.id,
    motion_id: motion.id,
  });
  try {
    const response = await fetch(`/api/feedback?${params}`);
    const result = await response.json();
    renderFeedbackHistory(result.items ?? []);
  } catch {
    elements.feedbackHistory.textContent = "反馈记录读取失败";
    elements.feedbackHistory.className = "feedback-history empty-copy";
  }
}

async function loadCatalog(refresh = false) {
  elements.catalogStatus.textContent = refresh ? "正在刷新动作数据" : "正在读取动作数据";
  elements.refreshButton.disabled = true;
  try {
    const response = await fetch(`/api/catalog${refresh ? "?refresh=1" : ""}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.catalog = await response.json();
    const backendExists = state.catalog.backends.some(
      (backend) => backend.id === state.backendId
    );
    if (!backendExists) state.backendId = state.catalog.backends[0]?.id ?? null;
    renderSelectors();
    elements.catalogStatus.textContent = `数据更新于 ${state.catalog.generated_at}`;
  } catch (error) {
    elements.catalogStatus.textContent = `数据读取失败: ${error.message}`;
  } finally {
    elements.refreshButton.disabled = false;
  }
}

async function submitFeedback(event) {
  event.preventDefault();
  const { backend, version, motion } = getSelection();
  const comment = elements.feedbackInput.value.trim();
  if (!backend || !version || !motion || !comment) return;

  elements.submitButton.disabled = true;
  elements.feedbackStatus.textContent = "正在发送";
  elements.feedbackStatus.className = "submit-status";
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        backend: backend.id,
        version: version.id,
        motion_id: motion.id,
        comment,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    elements.feedbackInput.value = "";
    elements.characterCount.textContent = "0 / 4000";
    elements.feedbackStatus.textContent = "已保存，等待下一轮 LLM 处理";
    elements.feedbackStatus.className = "submit-status success";
    await loadFeedbackHistory();
  } catch (error) {
    elements.feedbackStatus.textContent = error.message;
    elements.feedbackStatus.className = "submit-status error";
  } finally {
    elements.submitButton.disabled = false;
  }
}

elements.motionSelect.addEventListener("change", (event) => {
  state.motionId = event.target.value;
  state.versionId = null;
  renderSelectors();
});

elements.versionSelect.addEventListener("change", (event) => {
  state.versionId = event.target.value;
  renderSelection();
});

elements.refreshButton.addEventListener("click", () => loadCatalog(true));
elements.feedbackForm.addEventListener("submit", submitFeedback);
elements.feedbackInput.addEventListener("input", () => {
  elements.characterCount.textContent = `${elements.feedbackInput.value.length} / 4000`;
});

loadCatalog();
