// . Stereo / calibration parameters 
// Put your real calibration here.
// These numbers are from your K and stereo script.
const stereoParams = {
  fx: 1434.41,
  fy: 1430.68,
  cx: 949.77,
  cy: 541.41,
  baseline: 0.10 * (0.27 / 270.661),       // meters (distance between camera positions)
  disp_min_valid: 0.5625,
  disp_max_valid: 63.0
};

// 2. DOM elements
const leftInput   = document.getElementById("leftInput");
const dispInput   = document.getElementById("dispInput");
const picture     = document.getElementById("picture");
const aimer       = document.getElementById("aimer");
const dispCanvas  = document.getElementById("dispCanvas");
const resetBtn    = document.getElementById("resetBtn");
const statusP     = document.getElementById("status");

const aimerCtx = aimer.getContext("2d");
const dispCtx  = dispCanvas.getContext("2d");

// State
let imgLoaded  = false;
let dispLoaded = false;
let pictureCoords;

let points2D = [];   // [{u, v}]
let points3D = [];   // [{X, Y, Z}]

//  3. Load LEFT image 
leftInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const imgObj = new Image();
  imgObj.src = URL.createObjectURL(file);

  imgObj.onload = () => {
    // Put into the <img> element
    picture.src = imgObj.src;

    picture.onload = () => {
      const w = picture.naturalWidth;
      const h = picture.naturalHeight;

      // Match overlay canvas to image size
      aimer.width = w;
      aimer.height = h;

      // Update position info
      pictureCoords = picture.getBoundingClientRect();
      aimerCtx.clearRect(0, 0, w, h);

      imgLoaded = true;
      if (dispLoaded) {
        statusP.textContent =
          "Both images loaded. Click two points on the object to measure 3D distance.";
      } else {
        statusP.textContent =
          "Left image loaded. Now upload the disparity image for the SAME view.";
      }
    };
  };
});

// ---------- 4. Load DISPARITY image ----------
dispInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const dispImg = new Image();
  dispImg.src = URL.createObjectURL(file);

  dispImg.onload = () => {
    const w = dispImg.naturalWidth;
    const h = dispImg.naturalHeight;

    dispCanvas.width  = w;
    dispCanvas.height = h;

    // Draw disparity image into hidden canvas
    dispCtx.drawImage(dispImg, 0, 0, w, h);

    dispLoaded = true;
    if (imgLoaded) {
      statusP.textContent =
        "Both images loaded. Click two points on the object to measure 3D distance.";
    } else {
      statusP.textContent =
        "Disparity image loaded. Now upload the matching left image.";
    }
  };
});

// ---------- 5. Crosshair on mouse move ----------
aimer.addEventListener("mousemove", (e) => {
  if (!imgLoaded) return;

  pictureCoords = picture.getBoundingClientRect();
  const x = e.clientX - pictureCoords.left;
  const y = e.clientY - pictureCoords.top;

  aimerCtx.clearRect(0, 0, aimer.width, aimer.height);
  aimerCtx.beginPath();
  aimerCtx.strokeStyle = "#00ff00";
  aimerCtx.moveTo(0, y);
  aimerCtx.lineTo(aimer.width, y);
  aimerCtx.moveTo(x, 0);
  aimerCtx.lineTo(x, aimer.height);
  aimerCtx.stroke();
});

// ---------- 6. Click: choose points & compute 3D distance ----------
aimer.addEventListener("click", (e) => {
  if (!imgLoaded || !dispLoaded) {
    statusP.textContent = "Upload both left and disparity images first.";
    return;
  }

  pictureCoords = picture.getBoundingClientRect();
  const x = e.clientX - pictureCoords.left;
  const y = e.clientY - pictureCoords.top;

  // Map from displayed coords to image pixel coords
  const scaleX = picture.naturalWidth  / picture.clientWidth;
  const scaleY = picture.naturalHeight / picture.clientHeight;
  const u = x * scaleX;
  const v = y * scaleY;

  const P = pixelTo3D(u, v);
  if (!P) {
    statusP.textContent =
      "Invalid or zero disparity at this point – try another spot on the object.";
    return;
  }

  points2D.push({ u, v });
  points3D.push(P);

  drawPoint(u, v);

  if (points3D.length === 1) {
    statusP.textContent = "First point selected. Click a second point.";
  } else if (points3D.length === 2) {
    const d_m  = distance3D(points3D[0], points3D[1]); // meters
    const d_cm = d_m * 100.0;                          // centimeters
    drawLine(points2D[0], points2D[1]);

    const msg = `3D distance between points: ${d_cm.toFixed(2)} cm`;
    statusP.textContent = msg;
    alert(msg);  // make sure you see it

    // Next click will start a fresh pair
  } else {
    points2D = [];
    points3D = [];
    aimerCtx.clearRect(0, 0, aimer.width, aimer.height);
    statusP.textContent =
      "Resetting points. Click two points again to measure.";
  }
});

// ---------- 7. Pixel (u, v) -> 3D (X, Y, Z) using disparity ----------
function pixelTo3D(u, v) {
  const x = Math.round(u);
  const y = Math.round(v);

  const w = dispCanvas.width;
  const h = dispCanvas.height;
  if (x < 0 || x >= w || y < 0 || y >= h) return null;

  // disparity image is grayscale (R=G=B)
  const data = dispCtx.getImageData(x, y, 1, 1).data;
  const disp8 = data[0];   // 0..255

  const dmin = stereoParams.disp_min_valid;
  const dmax = stereoParams.disp_max_valid;
  const disp = dmin + (disp8 / 255.0) * (dmax - dmin);

  if (!isFinite(disp) || disp <= 0) return null;

  const { fx, fy, cx, cy, baseline: B } = stereoParams;

  const Z = fx * B / disp;       // meters
  const X = (u - cx) * Z / fx;
  const Y = (v - cy) * Z / fy;

  return { X, Y, Z };
}

// ---------- 8. 3D distance between two points ----------
function distance3D(p1, p2) {
  const dx = p1.X - p2.X;
  const dy = p1.Y - p2.Y;
  const dz = p1.Z - p2.Z;
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

// ---------- 9. Drawing helpers ----------
function drawPoint(u, v) {
  aimerCtx.beginPath();
  aimerCtx.arc(u, v, 6, 0, 2 * Math.PI);
  aimerCtx.fillStyle = "#ff0000";
  aimerCtx.fill();
}

function drawLine(p1, p2) {
  aimerCtx.beginPath();
  aimerCtx.strokeStyle = "#ffff00";
  aimerCtx.lineWidth = 2;
  aimerCtx.moveTo(p1.u, p1.v);
  aimerCtx.lineTo(p2.u, p2.v);
  aimerCtx.stroke();
}

// ---------- 10. Reset ----------
resetBtn.addEventListener("click", () => {
  points2D = [];
  points3D = [];
  aimerCtx.clearRect(0, 0, aimer.width, aimer.height);
  statusP.textContent =
    "Click two points on the object to measure 3D distance.";
});
