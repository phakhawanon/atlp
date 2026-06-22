/* ════════════════════════════════════════════════════════════════
   app.js  –  Video Inspector frontend logic
   
   BACKEND CONFIGURATION
   ─────────────────────
   Edit the constant below to point to your Python backend:
   ════════════════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:8000";   // ← EDIT: your Python backend URL


/* ════════════════════════════════════════════════════════════════
   VIDEO SYNC STATE
   ════════════════════════════════════════════════════════════════ */

const videos = [
  document.getElementById("video1"),
  document.getElementById("video2"),
  document.getElementById("video3"),
];

const overlays = [
  document.getElementById("overlay1"),
  document.getElementById("overlay2"),
  document.getElementById("overlay3"),
];

const isModelledButton = document.getElementById("is_modelled");
const isCheckedButton = document.getElementById("is_checked");
const modelTagButton = document.getElementById("model_tag");
const actualTagButton = document.getElementById("actual_tag");
const modelIsFailedButton = document.getElementById("model_is_failed");
const actualIsFailedButton = document.getElementById("actual_is_failed");
const saveButton = document.querySelector(".save-btn");

let filterOptions = {
  is_modelled: null,
  is_checked: null,
  model_tag: null,
  actual_tag: null,
  model_is_failed: null,
  actual_is_failed: null,
}

let modelTags = [];
let actualTags = [];
let currentModelTag = "";
let currentActualTag = "";
// // match button with filter option name
// filterButtonMatching = {
//   isModelledButton: "is_modelled",
//   isCheckedButton: "is_checked",
//   modelTagButton: "model_tag",
//   actualTagButton: "actual_tag",
//   modelIsFailedButton: "model_is_failed",
//   actualIsFailedButton: "actual_is_failed",
// }

function boolButtonSwitch(e) {
  // console.log("Button is clicked!");
  const targetButton = e.target;
  const targetOption = targetButton.id;
  // console.log(`targetOption is ${targetOption}`);
  // console.log(`targetButton is ${targetButton}`);
  const targetValue = filterOptions[targetOption];
  if (targetValue === null) {
    targetButton.classList.add("true");
    filterOptions[targetOption] = true;
  } else if (targetValue === true) {
    targetButton.classList.add("false");
    targetButton.classList.remove("true");
    filterOptions[targetOption] = false;
  } else if (targetValue === false) {
    targetButton.classList.remove("false");
    filterOptions[targetOption] = null;
  }
  handleDirectoryLoad();
}

function tagButtonSwitch(e) {
  const targetButton = e.target;
  const targetId = targetButton.id;
  let tagList, currentTag;
  if (targetId === "model_tag") {
    tagList = modelTags;
    currentTag = currentModelTag;
  } else {
    tagList = actualTags;
    currentTag = currentActualTag;
  }
  const tagNumber = tagList.length;
  let index = tagList.indexOf(currentTag);
  index = (index + 1) % tagNumber;
  if (targetId === "model_tag") {
    currentModelTag = tagList[index];
    targetButton.innerText = `model_tag=${tagList[index]}`;
    filterOptions.model_tag = currentModelTag;
  } else {
    currentActualTag = tagList[index];
    targetButton.innerText = `actual_tag=${tagList[index]}`;
    filterOptions.actual_tag = currentActualTag;
  }
  if (index === 0) {
    targetButton.classList.remove("occupied");
    if (targetId === "model_tag") {
      filterOptions.model_tag = null;
    } else {
      filterOptions.actual_tag = null;
    }
  }
  if (index > 0) {
    targetButton.classList.add("occupied");
  }
  handleDirectoryLoad();
}

function initFilterButton() {
  // console.log("initFilterButton() is runned.");
  isModelledButton.addEventListener("click", boolButtonSwitch);
  isCheckedButton.addEventListener("click", boolButtonSwitch);
  modelIsFailedButton.addEventListener("click", boolButtonSwitch);
  actualIsFailedButton.addEventListener("click", boolButtonSwitch);
}

function initTagButton() {
  modelTagButton.addEventListener("click", tagButtonSwitch);
  modelTagButton.classList.remove("occupied");
  modelTagButton.innerText = "model_tag=";
  actualTagButton.addEventListener("click", tagButtonSwitch);
  actualTagButton.classList.remove("occupied");
  actualTagButton.innerText = "actual_tag=";

}

const videoTitle = document.getElementById("video-title")

let isSyncPaused = false;   // true = all videos paused

/** Wire hover → pause/resume for every video card. */
function initVideoSync() {
  videos.forEach((vid, idx) => {
    const card = vid.closest(".video-card");

    card.addEventListener("mouseenter", () => {
      if (!isSyncPaused) {
        pauseAll();
        isSyncPaused = true;
        overlays[idx].textContent = "▶";
      }
    });

    card.addEventListener("mouseleave", () => {
      if (isSyncPaused) {
        playAll();
        isSyncPaused = false;
      }
    });

    // Also allow click to toggle while hovered
    card.addEventListener("click", () => {
      if (isSyncPaused) {
        playAll();
        isSyncPaused = false;
      } else {
        pauseAll();
        isSyncPaused = true;
      }
    });
  });
}

function playAll() {
  videos.forEach((v, i) => {
    v.play().catch(() => { });
    overlays[i].textContent = "⏸";
  });
}

function pauseAll() {
  videos.forEach((v, i) => {
    v.pause();
    overlays[i].textContent = "▶";
  });
}

/** Load a video URL into one of the three players (1-indexed). */
function loadVideo(index, path) {
  const vid = videos[index - 1];
  if (!vid) {
    console.log(`${path} at index ${index} is not a video.`);
    return;
  }
  console.log(`${path} at index ${index} is a video.`)
  // vid.src = url;
  vid.src = `${API_BASE}/video?path=${encodeURIComponent(path)}`;
  vid.load();
  vid.play().catch(() => { });
}


/* ════════════════════════════════════════════════════════════════
   DIRECTORY INPUT  →  BACKEND CALL
   ════════════════════════════════════════════════════════════════ */

const dirInput = document.getElementById("dir-input");

dirInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleDirectoryLoad(false, true);
});

const outputBox = document.getElementById("output-box");

async function handleDirectoryLoad(resetBool = false, resetTag = false) {

  const path = dirInput.value.trim();
  if (!path) {
    // appendOutput("⚠ Please enter a directory path.", "is-error");
    return;
  }

  clearOutput()

  if (resetTag === true) {
    console.log("resetTag is true");
    currentModelTag = "";
    currentActualTag = "";
    modelTags = [];
    actualTags = [];
  }

  try {
    /* ── START BACKEND CALL ── */
    const response = await fetch(`${API_BASE}/load-directory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, filterOptions }),
    });

    if (!response.ok) throw new Error(`Server error ${response.status}`);
    const data = await response.json();
    /* ── END BACKEND CALL ── */

    data.datapoints.forEach((datapoint) => {
      const entry = document.createElement('p');
      entry.classList.add("output-line")
      entry.textContent = datapoint;
      entry.addEventListener("click", handleDatapointLoad);
      outputBox.appendChild(entry);
      // console.log("Successfully append a datapoint");
    });

    if (resetTag === true) {
      modelTags = data.model_tag_list;
      actualTags = data.actual_tag_list;
      modelTags.unshift("");
      actualTags.unshift("");
      initTagButton();
    }

  } catch (err) {
    console.log(`Error loading directory! ${err.message}`);
  }
}

/**
 * Called when the user clicks "Load" or presses Enter.
 *
 * ── BACKEND HOOK ──────────────────────────────────────────────
 * Endpoint  : POST  {API_BASE}/load-directory
 * Request   : { "path": "<directory string>" }
 * Expected response (JSON):
 *   {
 *     "output":  ["line1", "line2", ...],   // strings for the Output log
 *     "video1":  "/static/vid1.mp4",        // URL or relative path
 *     "video2":  "/static/vid2.mp4",
 *     "video3":  "/static/vid3.mp4",
 *     "json1":   { ... },                   // object for JSON editor 1
 *     "json2":   { ... }                    // object for JSON editor 2
 *   }
 * ──────────────────────────────────────────────────────────────
 */

let loadedDatapoint = null;

async function handleDatapointLoad(e) {

  const path = dirInput.value.trim();

  if (loadedDatapoint !== null) {
    loadedDatapoint.classList.remove("loaded-datapoint");
    loadedDatapoint.addEventListener("click", handleDatapointLoad);
  }

  loadedDatapoint = e.target;
  loadedDatapoint.classList.add("loaded-datapoint");
  loadedDatapoint.removeEventListener("click", handleDatapointLoad);
  console.log("Frontend receive datapoint load request.");
  datapoint = loadedDatapoint.textContent;
  videoTitle.textContent = `Videos from ${datapoint}`;

  try {
    /* ── START BACKEND CALL ── */
    const response = await fetch(`${API_BASE}/load-datapoint`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, datapoint }),
    });

    if (!response.ok) throw new Error(`Server error ${response.status}`);
    const data = await response.json();
    /* ── END BACKEND CALL ── */

    // // Populate output log
    // if (Array.isArray(data.output)) {
    //   data.output.forEach(line => appendOutput(line));
    // }

    // console.log("running1");
    // console.log(data.video1);
    // console.log(data.video2);
    // console.log(data.video3);
    // console.log("running2");
    // Load videos
    if (data.video1) loadVideo(1, data.video1);
    // console.log("running3");
    if (data.video2) loadVideo(2, data.video2);
    // console.log("running4");
    if (data.video3) loadVideo(3, data.video3);

    // Load JSON editors
    if (data.json1) loadJson(1, data.json1);
    if (data.json2) loadJson(2, data.json2);

    // appendOutput("✔ Loaded successfully.", "is-success");

  } catch (err) {
    // appendOutput(`✘ ${err.message}`, "is-error");
    console.error("handleDatapointLoad error:", err);
  }
  saveButton.innerText = "Save";
  // saveButton.classList.remove("false");
}


/* ════════════════════════════════════════════════════════════════
   OUTPUT LOG
   ════════════════════════════════════════════════════════════════ */


function appendOutput(text, cssClass = "") {
  // Remove placeholder on first real entry
  const placeholder = outputBox.querySelector(".placeholder-text");
  if (placeholder) placeholder.remove();

  const line = document.createElement("span");
  line.className = `output-line ${cssClass}`.trim();
  line.textContent = text;
  outputBox.appendChild(line);
  outputBox.scrollTop = outputBox.scrollHeight;
}

function clearOutput() {
  outputBox.innerHTML = '';
}


/* ════════════════════════════════════════════════════════════════
   JSON EDITORS
   ════════════════════════════════════════════════════════════════ */

// Internal store so we can rebuild before saving
const jsonStore = { 1: {}, 2: {} };

/**
 * Render a JSON object into the editable field grid.
 * @param {1|2}  index  – which editor (1 or 2)
 * @param {object} data – plain JS object from the backend
 */
function loadJson(index, data) {
  const container = document.getElementById(`json${index}-editor`);
  container.innerHTML = "";
  jsonStore[index] = JSON.parse(JSON.stringify(data));  // deep copy
  renderFields(container, data, jsonStore[index]);
}

/**
 * Recursively render key-value pairs as labelled textareas.
 * Nested objects get an indented sub-section.
 */
function renderFields(container, source, store) {
  // console.log(source);
  if (!source || typeof source !== "object") {
    container.innerHTML = '<p class="json-placeholder">No data.</p>';
    return;
  }

  Object.entries(source).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "json-field";

    const keyEl = document.createElement("div");
    keyEl.className = "json-key";
    keyEl.textContent = key;

    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      // Nested object → recurse into a sub-container
      const nested = document.createElement("div");
      nested.className = "json-nested";
      store[key] = JSON.parse(JSON.stringify(value));
      renderFields(nested, value, store[key]);
      row.appendChild(keyEl);
      row.appendChild(nested);
    } else {
      // Scalar / array value → editable textarea
      const valEl = document.createElement("textarea");
      valEl.className = "json-value";
      valEl.value = Array.isArray(value) ? JSON.stringify(value) : String(value ?? "");
      valEl.rows = 1;
      valEl.spellcheck = false;

      // Auto-expand height
      const autoResize = () => {
        valEl.style.height = "auto";
        valEl.style.height = valEl.scrollHeight + "px";
      };
      valEl.addEventListener("input", () => {
        store[key] = valEl.value;
        autoResize();
      });
      setTimeout(autoResize, 0);

      row.appendChild(keyEl);
      row.appendChild(valEl);
    }

    container.appendChild(row);
  });
}

/**
 * Collect current editor values and POST to backend.
 *
 * ── BACKEND HOOK ──────────────────────────────────────────────
 * Endpoint  : POST  {API_BASE}/save-json/{index}
 * Request   : { "data": { ...current field values... } }
 * Expected response (JSON):
 *   { "status": "ok" }   or   { "error": "message" }
 * ──────────────────────────────────────────────────────────────
 */
async function saveJson() {
  // const payload1 = JSON.stringify(jsonStore[1]);
  // const payload2 = JSON.stringify(jsonStore[2]);
  const payload1 = jsonStore[1];
  const payload2 = jsonStore[2];
  console.log(payload2);
  // appendOutput(`Saving JSON ${index}…`);

  try {
    /* ── START BACKEND CALL ── */
    const response = await fetch(`${API_BASE}/save-json/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_data: payload1,
        actual_data: payload2,
        datapoint: loadedDatapoint.textContent
      }),
    });

    if (!response.ok) throw new Error(`Server error ${response.status}`);
    const result = await response.json();
    /* ── END BACKEND CALL ── */

    if (result.error) throw new Error(result.error);
    // appendOutput(`✔ JSON ${index} saved.`, "is-success");

    handleDirectoryLoad(false, true)
    saveButton.innerText = "Saved!";
    // saveButton.classList.remove("false");

  } catch (err) {
    // appendOutput(`✘ Save JSON ${index}: ${err.message}`, "is-error");
    console.error(`saveJson() error:`, err);
    saveButton.innerText = "Save Fail!";
    // saveButton.classList.add("false");
    // saveButton.classList.remove("true");
  }
}


/* ════════════════════════════════════════════════════════════════
   INIT
   ════════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  // console.log("Hello World!")
  // console.log("HTML is parsed");
  initVideoSync();
  initFilterButton();
  // console.log("Hello World!");
  // ── OPTIONAL: pre-populate JSON editors with mock data for UI testing ──
  // Remove or comment out these lines once your backend is connected.
  // loadJson(1, {
  //   name: "clip_001",
  //   fps: "30",
  //   duration: "12.4",
  //   tags: ["outdoor", "day"],
  //   metadata: { camera: "GoPro 12", resolution: "4K" }
  // });

  // loadJson(2, {
  //   label: "scene_A",
  //   confidence: "0.94",
  //   objects: ["person", "car"],
  //   bbox: { x: "120", y: "80", w: "200", h: "150" }
  // });

  // ── OPTIONAL: Start playback automatically if you pre-load video sources ──
  // playAll();
});
