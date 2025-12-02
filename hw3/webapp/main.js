function setupForm(formId, url, statusId) {
    const form = document.getElementById(formId);
    const statusEl = document.getElementById(statusId);
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        statusEl.classList.remove("error", "success");
        statusEl.textContent = "Uploading and processing...";
        const formData = new FormData(form);
        try {
            const resp = await fetch(url, { method: "POST", body: formData });
            const data = await resp.json();
            if (!data.success) {
                statusEl.classList.add("error");
                statusEl.textContent = data.message || "Something went wrong.";
                return;
            }
            statusEl.classList.add("success");
            let msg = data.message || "Done.";
            if (data.output_dir) {
                msg += ` Outputs saved in: ${data.output_dir}`;
            }
            statusEl.textContent = msg;
        } catch (err) {
            console.error(err);
            statusEl.classList.add("error");
            statusEl.textContent = "Request failed (see console).";
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupForm("form-part1", "/hw3/run_part1", "status-part1");
    setupForm("form-part2", "/hw3/run_part2", "status-part2");
    setupForm("form-part3", "/hw3/run_part3", "status-part3");
    setupForm("form-part4", "/hw3/run_part4", "status-part4");
    setupForm("form-part5", "/hw3/run_part5", "status-part5");
});


