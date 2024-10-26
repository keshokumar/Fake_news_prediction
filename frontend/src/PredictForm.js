import React, { useState } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
    const [inputText, setInputText] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [probability, setProbability] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: inputText }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            setPrediction(data.prediction);
            setProbability(data.probability);
        } catch (error) {
            console.error("Error fetching prediction:", error);
            setPrediction(null);
            setProbability(null);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1 className="app-title">Fake News Detection</h1>
                <p className="info-text">
                    In today's world, distinguishing between true and false information is crucial. 
                    Enter news text to check its authenticity.
                </p>
                <form onSubmit={handleSubmit} className="input-form">
                    <textarea
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Enter news text here"
                        rows="4"
                        className="form-control"
                        required
                    ></textarea>
                    <button type="submit" className="btn btn-primary submit-button">Submit</button>
                </form>
                {prediction !== null && (
                    <div className="alert alert-info result-alert">
                        {prediction === 1 ? 'The news is likely TRUE!' : 'The news is likely FAKE!'}
                        <br />
                        Probability: {probability !== null ? probability.toFixed(2) : 'N/A'}
                    </div>
                )}
            </header>
        </div>
    );
}

export default App;
