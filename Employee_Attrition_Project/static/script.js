
const fileInput = document.getElementById("file");
const fileName = document.getElementById("fileName");
const fileInfo = document.getElementById("fileInfo");
const uploadForm = document.getElementById("uploadForm");

fileInput.addEventListener("change", function () {
    const selectedFile = this.files[0];

    if (selectedFile) {
        fileName.textContent = selectedFile.name;

        const sizeKB = (selectedFile.size / 1024).toFixed(1);
        fileInfo.textContent = `${sizeKB} KB — ready to upload`;
    } else {
        fileName.textContent = "Choose your CSV file";
        fileInfo.textContent = "Click to browse your computer";
    }
});

uploadForm.addEventListener("submit", function (event) {
    const selectedFile = fileInput.files[0];

    if (!selectedFile) {
        event.preventDefault();
        alert("Please select a CSV file.");
        return;
    }

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
        event.preventDefault();
        alert("Please select a .csv file.");
    }
});
