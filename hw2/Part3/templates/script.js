function uploadImage() {
    let fileInput = document.getElementById("imageInput");

    if (fileInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }

    let file = fileInput.files[0];

    document.getElementById("originalImage").src = URL.createObjectURL(file);

    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData,
    })
    .then(res => res.json())
    .then(data => {
        if (data.output_url) {
            document.getElementById("outputImage").src = data.output_url;
        }
    })
    .catch(err => console.error(err));
}
