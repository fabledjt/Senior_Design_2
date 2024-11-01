import React, { useEffect, useState, useRef } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./MenuPage.css"; // Import the CSS file

function MenuPage() {
  const location = useLocation();
  const { userAnswers } = location.state || {};
  const navigate = useNavigate();
  const [isSaving, setIsSaving] = useState(false); // State to track saving status
  const hasSavedRef = useRef(false); // Ref to track if answers are already saved

  useEffect(() => {
    console.log("User Answers:", userAnswers); // Log userAnswers for debugging

    const saveAnswers = async () => {
      if (hasSavedRef.current || isSaving) return; // Prevent multiple submissions
      hasSavedRef.current = true; // Mark as saved
      setIsSaving(true); // Set saving status

      if (userAnswers && userAnswers.length > 0) {
        try {
          const response = await fetch("http://localhost:5000/api/saveAnswers", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(
              userAnswers.map((answer) => ({
                imageName: answer.image.split('/').pop(),
                userAnswer: answer.userAnswer,
                correctAnswer: answer.correctAnswer,
                isCorrect: answer.userAnswer === answer.correctAnswer,
              }))
            ),
          });

          const responseData = await response.json(); // Get response data
          console.log("Response from server:", response); // Log the response object

          if (!response.ok) {
            console.error("Error response:", responseData); // Log error details
            throw new Error("Network response was not ok");
          }

          console.log("Answers saved successfully:", responseData);
        } catch (error) {
          console.error("Error saving answers:", error);
        } finally {
          setIsSaving(false); // Reset saving status
        }
      } else {
        console.warn("No user answers to save."); // Log if no answers
        setIsSaving(false); // Reset saving status
      }
    };

    // Only call saveAnswers when the component mounts or userAnswers change
    if (userAnswers && userAnswers.length > 0) {
      saveAnswers();
    }
  }, [userAnswers]); // No isSaving or hasSavedRef in dependency array

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
