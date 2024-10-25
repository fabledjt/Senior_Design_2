import React, { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./MenuPage.css"; // Import the CSS file

function MenuPage() {
  const location = useLocation();
  const { userAnswers } = location.state || [];
  const navigate = useNavigate();

  useEffect(() => {
    if (userAnswers && userAnswers.length > 0) {
      const textContent = userAnswers.map((answer, index) => {
        return `Answer ${index + 1}:
Your Answer: ${answer.userAnswer}
Correct Answer: ${answer.correctAnswer}
${answer.userAnswer === answer.correctAnswer ? "Correct!" : "Wrong!"}
----------------------------------------`;
      }).join("\n");

      const blob = new Blob([textContent], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "UserAnswers.txt";
      link.click();
      URL.revokeObjectURL(url);
    }
  }, [userAnswers]);

  const handleRestartQuiz = () => {
    navigate("/quiz", { state: { userAnswers } });
  };
  
  const handleReturnStart = () => {
    navigate("/");
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
        <button className="menureturn-button" onClick={handleReturnStart}>
          Go to Start Quiz
        </button>
      </div>
    </div>
  );
}

export default MenuPage;
