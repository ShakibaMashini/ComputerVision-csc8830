// ---------- Part 1 ----------
document.getElementById("form_part1").addEventListener("submit", function(e){
    e.preventDefault();

    let formData = new FormData(this);

    fetch("/process_part1", {
        method: "POST",
        body: formData
    })
    .then(r => r.text())
    .then(path => {
        document.getElementById("part1_result").innerHTML =
            `<h3>Panorama Result</h3>
             <img src="/${path}?t=${Date.now()}">`;
    });
});


// ---------- Part 2 ----------
document.getElementById("form_part2").addEventListener("submit", function(e){
    e.preventDefault();

    let formData = new FormData(this);

    fetch("/process_part2", {
        method: "POST",
        body: formData
    })
    .then(r => r.text())
    .then(text => {
        let parts = text.split("||");
        let ours = parts[0];
        let cv = parts[1];

        document.getElementById("part2_result").innerHTML =
            `
            <h3>Our SIFT Implementation</h3>
            <img src="/${ours}?t=${Date.now()}">

            <h3>OpenCV SIFT</h3>
            <img src="/${cv}?t=${Date.now()}">
            `;
    });
});
