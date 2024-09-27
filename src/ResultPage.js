import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./ResultPage.css"; // Import the CSS file

function ResultPage() {
  const location = useLocation();
  const { userAnswers } = location.state || [];
  const navigate = useNavigate();

  const handleLoadAnotherQuestion = () => {
    navigate("/quiz", { state: { userAnswers } });
  };

  const handleViewPastAnswers = () => {
    navigate("/menu", { state: { userAnswers } });
  };

  return (
    <div className="result-container">
      <nav className="navbar">
        <h2>Quiz Navigation</h2>
      </nav>
      <div className="content">
        <h1>Quiz Results</h1>
        {userAnswers && userAnswers.length > 0 ? (
          <div>
            <p>Your most recent answer:</p>
            <ul>
              {userAnswers.slice(-1).map((answer, index) => (
                <li key={index}>
                  <p>
                    Image: <img src={answer.image} alt={`Answer ${index}`} width={100} />
                  </p>
                  <p>Your Answer: {answer.userAnswer}</p>
                  <p>Correct Answer: {answer.correctAnswer}</p>
                  <p>{answer.userAnswer === answer.correctAnswer ? "Correct!" : "Wrong!"}</p>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p>No results to display.</p>
        )}
        <div className="button-group">
          <button className="result-button" onClick={handleLoadAnotherQuestion}>
            Load Another Question
          </button>
          <button className="result-button" onClick={handleViewPastAnswers}>
            View Past Answers
          </button>
        </div>
      </div>
    </div>
  );
}

export default ResultPage;
