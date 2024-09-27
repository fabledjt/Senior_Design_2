import React from "react";
import { useNavigate } from "react-router-dom";
import "./StartQuiz.css"; // Import the CSS file

function StartQuiz() {
  const navigate = useNavigate();

  const startQuiz = () => {
    navigate("/quiz");
  };

  return (
    <div className="start-container">
      <nav className="navbar">
        <h2>Quiz Navigation</h2>
      </nav>
      <div className="content">
        <h1>Welcome to the Cheese or Bread Quiz</h1>
        <button className="start-button" onClick={startQuiz}>Start Quiz</button>
      </div>
    </div>
  );
}

export default StartQuiz;
