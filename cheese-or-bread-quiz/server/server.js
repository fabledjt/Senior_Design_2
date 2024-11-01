const express = require("express");
const fs = require("fs");
const path = require("path");
const cors = require("cors"); // Import the CORS package
const app = express();

const corsOptions = {
  origin: "http://localhost:3000", // Replace with your React app's URL if different
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ["Content-Type"],
};

app.use(cors(corsOptions));

app.use(express.json());

app.post("/api/saveAnswers", (req, res) => {
  console.log("Received answers:", req.body); // Log the received answers
  const answers = req.body;

  // Determine the directory for answers
  const dir = path.join(__dirname, "answers");
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }

  // Determine the file name based on existing files
  let fileIndex = 1;
  let fileName = path.join(dir, `answers${fileIndex}.json`);
  
  while (fs.existsSync(fileName)) {
    fileIndex += 1;
    fileName = path.join(dir, `answers${fileIndex}.json`);
  }

  console.log("Saving to directory:", dir); // Log directory
  console.log("File name:", fileName); // Log file name

  // Write answers to a new JSON file
  fs.writeFile(fileName, JSON.stringify(answers, null, 2), (err) => {
    if (err) {
      console.error("Error writing to file:", err);
      return res.status(500).json({ message: "Error saving answers." });
    }
    console.log("Answers saved to", fileName);
    res.status(200).json({ message: "Answers saved successfully." });
  });
});

// Error handling for any unhandled errors
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ message: "Internal server error." });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
