import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./MenuPage.css"; // Import the CSS file

function MenuPage() {
  const location = useLocation();
  const { userAnswers } = location.state || [];
  const navigate = useNavigate();

  const handleRestartQuiz = () => {
    navigate("/quiz", { state: { userAnswers } }); // Pass userAnswers to the quiz page if needed
  };
  
  const handleReturnStart = () => {
    navigate("/"); // Navigate back to the start page to restart the quiz
  };

  return (
    <div className="menu-container">
      <div className="content">
        <h1>All Past Answers</h1>
        {userAnswers && userAnswers.length > 0 ? (
          <ul>
            {userAnswers.map((answer, index) => (
              <li key={index}>
                <p>
                  Image: <img src={answer.image} alt={`Answer ${index}`} width={100} />
                </p>
                <p>Your Answer: {answer.userAnswer}</p>
                <p>Correct Answer: {answer.correctAnswer}</p>
                <p>{answer.userAnswer === answer.correctAnswer ? "Correct!" : "Wrong!"}</p>
                <hr />
              </li>
            ))}
          </ul>
        ) : (
          <p>No past answers available.</p>
        )}
        <button className="restart-button" onClick={handleRestartQuiz}>
          Restart Quiz
        </button>
        <h4><button className="menureturn-button" onClick={handleReturnStart}>
          Go to Start Quiz
        </button></h4>
      </div>
    </div>
  );
}

export default MenuPage;
