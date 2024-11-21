// client/src/pages/QuizPage.js
import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "./QuizPage.css"; // Import the CSS file

const images = [
  { src: "/images/bread1.png", type: "bread" },
  { src: "/images/bread2.png", type: "bread" },
  { src: "/images/bread3.png", type: "bread" },
  { src: "/images/bread4.png", type: "bread" },
  { src: "/images/bread5.png", type: "bread" },
  { src: "/images/bread6.png", type: "bread" },
  { src: "/images/bread7.png", type: "bread" },
  { src: "/images/bread8.png", type: "bread" },
  { src: "/images/bread9.png", type: "bread" },
  { src: "/images/bread10.png", type: "bread" },
  { src: "/images/bread11.png", type: "bread" },
  // ... add all your images here
  { src: "/images/cheese1.png", type: "cheese" },
  { src: "/images/cheese2.png", type: "cheese" },
  { src: "/images/cheese3.png", type: "cheese" },
  { src: "/images/cheese4.png", type: "cheese" },
  { src: "/images/cheese5.png", type: "cheese" },
  { src: "/images/cheese6.png", type: "cheese" },
  { src: "/images/cheese7.png", type: "cheese" },
  { src: "/images/cheese8.png", type: "cheese" },
  { src: "/images/cheese9.png", type: "cheese" },
  { src: "/images/cheese10.png", type: "cheese" },
  { src: "/images/cheese11.png", type: "cheese" },
  { src: "/images/cheese12.png", type: "cheese" },
];

function QuizPage() {
  const [currentImage, setCurrentImage] = useState(null);
  const navigate = useNavigate();
  const location = useLocation();
  const [userAnswers, setUserAnswers] = useState(location.state?.userAnswers || []);

  useEffect(() => {
    selectRandomImage();
  }, []);

  const selectRandomImage = () => {
    const randomImage = images[Math.floor(Math.random() * images.length)];
    setCurrentImage(randomImage);
  };

  const handleAnswer = (answer) => {
    const newAnswer = {
      image: currentImage.src,
      correctAnswer: currentImage.type,
      userAnswer: answer,
    };
    setUserAnswers((prevAnswers) => [...prevAnswers, newAnswer]);
    selectRandomImage();
  };

  const handleFinishQuiz = async () => {
    navigate("/menu", { state: { userAnswers } });
  };

  const handleRestartQuiz = () => {
    navigate("/"); // Navigate back to the start page to restart the quiz
  };

  if (!currentImage) return <div>Loading...</div>;

  return (
    <div className="quiz-container">
      <div className="content">
        <h1>Is this Cheese or Bread?</h1>
        <img src={currentImage.src} alt="Quiz" width={300} className="quiz-image" />
        <div className="button-group">
          <button className="quiz-button" onClick={() => handleAnswer("bread")}>Bread</button>
          <button className="quiz-button" onClick={() => handleAnswer("cheese")}>Cheese</button>
        </div>
        <button className="done-button" onClick={handleFinishQuiz}>Done</button>
        <h4>
          <button className="quizreturn-button" onClick={handleRestartQuiz}>
            Go to Start Quiz
          </button>
        </h4>
      </div>
    </div>
  );
}

export default QuizPage;
