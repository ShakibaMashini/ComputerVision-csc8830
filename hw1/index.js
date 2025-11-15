let K = [
    [1434.41, 0, 949.77],
    [0, 1430.68, 541.41],
    [0, 0, 1]
  ];
  K_inv = math.inv(K);
  let org_w, org_h;
  let picture = document.getElementById("picture");
  let pictureCoords = picture.getBoundingClientRect();
  let ruler = document.getElementById("ruler");
  let aimer = document.getElementById("aimer");
  let rulerCtx = ruler.getContext("2d");
  let aimerCtx = aimer.getContext("2d");
  
  function upload(e) {
    let file = e.target.files[0];
    let org_image = new Image();
    org_image.src = URL.createObjectURL(file);
    org_image.onload = () => { 
      org_w = org_image.width;
      org_h = org_image.height;
    }
    picture.src = org_image.src;
    picture.onload = () => {
      ruler.width = picture.width;
      ruler.height = picture.height;
      aimer.width = picture.width;
      aimer.height = picture.height;
      pictureCoords = picture.getBoundingClientRect();
    };
  }
  
  function showAim(e) {
    aimerCtx.clearRect(0, 0, aimer.width, aimer.height);
    let { clientX, clientY } = e;
    let x = clientX - pictureCoords.left;
    let y = clientY - pictureCoords.top;
  
    aimerCtx.beginPath();
    aimerCtx.strokeStyle = "#00ff00";
    aimerCtx.moveTo(0, y);
    aimerCtx.lineTo(aimer.width, y);
    aimerCtx.moveTo(x, 0);
    aimerCtx.lineTo(x, aimer.height);
    aimerCtx.stroke();
  }
  ruler.onmousemove = showAim;
  
  function point(x, y) {
    rulerCtx.beginPath();
    rulerCtx.arc(x, y, 7, 0, 2 * Math.PI);
    rulerCtx.fillStyle = "#00ff00";
    rulerCtx.fill();
  }
  
  let one, two, zc;
  let firstClick = true;
  function getCoords(e) {
    let { clientX, clientY } = e;
    let x = (clientX - pictureCoords.left);
    let y = (clientY - pictureCoords.top);
    let scaled_x = x * org_w / picture.width;
    let scaled_y = y * org_h / picture.height;
    if (firstClick) {
      rulerCtx.clearRect(0, 0, aimer.width, aimer.height);
      one = [[scaled_x], [scaled_y], [1]];
      point(x, y);
    } else {
      two = [[scaled_x], [scaled_y], [1]];
      point(x, y);
      zc = parseFloat(prompt('Distance to camera?'));
      alert(`The length ${computeDistance(one, two).toFixed(2)}`);
    }
    firstClick = !firstClick;
  }
  ruler.onclick = getCoords;
  
  function computeDistance(one, two) {
    one_w = math.multiply(K_inv, one)
    console.log(one_w);
    two_w = math.multiply(K_inv, two)
    console.log(two_w);
    one_w = one_w.map(row => row.map(element => element * zc))
    console.log(one_w);
    two_w = two_w.map(row => row.map(element => element * zc))
    console.log(two_w);
    d = Math.sqrt(
      Math.pow(one_w[0][0] - two_w[0][0], 2) +
      Math.pow(one_w[1][0] - two_w[1][0], 2) +
      Math.pow(one_w[2][0] - two_w[2][0], 2)
    )
    return d;
  }