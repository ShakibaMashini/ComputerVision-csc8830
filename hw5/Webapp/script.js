// Buttons
const btnAruco = document.getElementById("btn-aruco");
const btnMarkerless = document.getElementById("btn-markerless");
const btnSam2 = document.getElementById("btn-sam2");
const btnMakeNpz = document.getElementById("btn-make-npz");
const npzStatus = document.getElementById("npz-status");

// Canvas for drawing rectangle
const imgInput = document.getElementById("sam-image");
const imgEl = document.getElementById("sam-img");
const canvas = document.getElementById("sam-canvas");
const ctx = canvas.getContext("2d");

let drawing = false;
let startX = 0;
let startY = 0;
let endX = 0;
let endY = 0;
let currentNpzPath = null; // server-side path

btnAruco.addEventListener("click", async () => {
  await fetch("/hw5/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "aruco" })
  });
  alert("ArUco tracker launched. Check the OpenCV window; press ESC to quit.");
});

btnMarkerless.addEventListener("click", async () => {
  await fetch("/hw5/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "markerless" })
  });
  alert("Markerless tracker launched. Check the OpenCV window; press 's' to select ROI, ESC to quit.");
});

btnSam2.addEventListener("click", async () => {
  if (!currentNpzPath) {
    alert("Create NPZ first.");
    return;
  }
  await fetch("/hw5/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "sam2", npz_path: currentNpzPath })
  });
  alert("SAM2 tracker launched with your NPZ. Check the OpenCV window; press ESC to quit.");
});

imgInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  imgEl.onload = () => {
    canvas.width = imgEl.naturalWidth;
    canvas.height = imgEl.naturalHeight;
    drawImage();
    btnMakeNpz.disabled = false;
  };
});

function drawImage() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
  if (drawing || (endX > 0 && endY > 0)) {
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);
    const w = Math.abs(endX - startX);
    const h = Math.abs(endY - startY);
    ctx.strokeStyle = "#00e6ac";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
  }
}

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  startX = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
  startY = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));
  endX = startX;
  endY = startY;
  drawing = true;
  drawImage();
});

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  endX = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
  endY = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));
  drawImage();
});

canvas.addEventListener("mouseup", () => {
  drawing = false;
  drawImage();
});

btnMakeNpz.addEventListener("click", async () => {
  const x = Math.min(startX, endX);
  const y = Math.min(startY, endY);
  const w = Math.abs(endX - startX);
  const h = Math.abs(endY - startY);
  if (w <= 0 || h <= 0) {
    alert("Draw a rectangle over the object first.");
    return;
  }
  npzStatus.textContent = "Creating NPZ...";
  try {
    const resp = await fetch("/hw5/make_npz", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x, y, w, h })
    });
    const data = await resp.json();
    if (!data.success) {
      npzStatus.textContent = data.message || "Failed to create NPZ.";
      return;
    }
    currentNpzPath = data.npz_path;
    btnSam2.disabled = false;
    npzStatus.innerHTML = `NPZ ready: <a href="${data.npz_url}" target="_blank">download</a>`;
  } catch (e) {
    console.error(e);
    npzStatus.textContent = "Error creating NPZ.";
  }
});


