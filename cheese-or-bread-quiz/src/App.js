import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import StartQuiz from "./StartQuiz";
import QuizPage from "./QuizPage";
import MenuPage from "./MenuPage";  // Import the new MenuPage

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<StartQuiz />} />
          <Route path="/quiz" element={<QuizPage />} />
          <Route path="/menu" element={<MenuPage />} /> {/* New route for MenuPage */}
        </Routes>
      </div>
    </Router>
  );
}

export default App;
