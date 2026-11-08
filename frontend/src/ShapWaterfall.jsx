
/*
============================================================
DIAPREDICT - INDIVIDUAL SHAP WATERFALL PLOT
============================================================

PURPOSE:
1. Retrieve the user's prediction history from FastAPI.
2. Allow selection of a previous prediction.
3. Retrieve the selected prediction's SHAP explanation.
4. Display the base value and individual contributions.
5. Calculate the cumulative model output.
6. Display the final model output.

DATA SOURCE:
GET /predictions/history
GET /predictions/{prediction_id}/explanation

IMPORTANT:
All chart values come from the actual backend API.
SHAP values and model output are displayed in log-odds space.

============================================================
*/

// Import React hooks for state management and API requests.
import { useEffect, useState } from 'react';
import { API_BASE_URL } from './config';

// ==========================================================
// 1. SHAP WATERFALL COMPONENT
// ==========================================================

function ShapWaterfall({ authToken, initialPredictionId = '', lockSelection = false }) {

  // Store all predictions belonging to the logged-in user.
  const [predictionHistory, setPredictionHistory] = useState([]);

  // Store the ID of the prediction selected by the user.
  const [selectedPredictionId, setSelectedPredictionId] = useState(String(initialPredictionId || ''));

  // Store the individual SHAP explanation returned by FastAPI.
  const [explanation, setExplanation] = useState(null);

  // Track the prediction history loading state.
  const [historyLoading, setHistoryLoading] = useState(true);

  // Track the individual explanation loading state.
  const [explanationLoading, setExplanationLoading] = useState(false);

  // Store prediction history API errors.
  const [historyError, setHistoryError] = useState('');

  // Store individual explanation API errors.
  const [explanationError, setExplanationError] = useState('');


  // ========================================================
  // 2. FETCH PREDICTION HISTORY
  // ========================================================

  useEffect(() => {

    // Do not request data without an authentication token.
    if (!authToken) {
      setHistoryError('Please log in to view prediction explanations.');
      setHistoryLoading(false);
      return;
    }

    // Create a controller to cancel the request if necessary.
    const controller = new AbortController();

    // Define the asynchronous history request.
    const fetchPredictionHistory = async () => {

      // Start loading and clear any previous error.
      setHistoryLoading(true);
      setHistoryError('');

      try {

        // Request the authenticated user's prediction history.
        const response = await fetch(
          `${API_BASE_URL}/predictions/history`,
          {
            method: 'GET',

            // Send the JWT token to authenticate the request.
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json',
            },

            // Allow the request to be cancelled.
            signal: controller.signal,
          }
        );

        // Convert the response into JavaScript data.
        const data = await response.json();

        // Handle unsuccessful HTTP responses.
        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load prediction history.'
          );
        }

        // Verify that the API returned an array.
        if (!Array.isArray(data)) {
          throw new Error('Invalid prediction history response.');
        }

        // Store the actual prediction history.
        setPredictionHistory(data);

        // Select the requested prediction when supplied; otherwise use the latest.
        if (data.length > 0) {
          const requestedRecord = initialPredictionId
            ? data.find((record) => String(record.id) === String(initialPredictionId))
            : null;

          setSelectedPredictionId(
            requestedRecord ? String(requestedRecord.id) : String(data[0].id)
          );
        }

      } catch (error) {

        // Ignore errors caused by request cancellation.
        if (error.name === 'AbortError') {
          return;
        }

        // Display the actual API error.
        setHistoryError(error.message);

      } finally {

        // Stop loading if the request was not cancelled.
        if (!controller.signal.aborted) {
          setHistoryLoading(false);
        }
      }
    };

    // Execute the history request.
    fetchPredictionHistory();

    // Cancel the request when the component is removed.
    return () => {
      controller.abort();
    };

  }, [authToken, initialPredictionId]);


  // ========================================================
  // 3. FETCH SELECTED PREDICTION EXPLANATION
  // ========================================================

  useEffect(() => {

    // Stop if no prediction has been selected.
    if (!selectedPredictionId || !authToken) {
      setExplanation(null);
      return;
    }

    // Create a cancellation controller for this request.
    const controller = new AbortController();

    // Define the individual explanation request.
    const fetchExplanation = async () => {

      // Start explanation loading.
      setExplanationLoading(true);

      // Clear the previous explanation and error.
      setExplanation(null);
      setExplanationError('');

      try {

        // Build the endpoint using the selected prediction ID.
        const endpoint =
          `${API_BASE_URL}/predictions/${encodeURIComponent(selectedPredictionId)}/explanation`;

        // Request the selected prediction's SHAP explanation.
        const response = await fetch(endpoint, {
          method: 'GET',

          // Authenticate the request using the JWT token.
          headers: {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },

          // Allow cancellation when selection changes.
          signal: controller.signal,
        });

        // Convert the response into JavaScript data.
        const data = await response.json();

        // Handle unsuccessful responses.
        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load SHAP explanation.'
          );
        }

        // Verify the required explanation fields.
        if (
          typeof data.base_value !== 'number' ||
          typeof data.model_output !== 'number' ||
          !data.feature_contributions ||
          typeof data.feature_contributions !== 'object'
        ) {
          throw new Error('Invalid SHAP explanation response.');
        }

        // Store the actual explanation from FastAPI.
        setExplanation(data);

      } catch (error) {

        // Ignore intentional request cancellation.
        if (error.name === 'AbortError') {
          return;
        }

        // Display the actual API error.
        setExplanationError(error.message);

      } finally {

        // Stop loading only if the request was not cancelled.
        if (!controller.signal.aborted) {
          setExplanationLoading(false);
        }
      }
    };

    // Execute the explanation request.
    fetchExplanation();

    // Cancel the previous request when selection changes.
    return () => {
      controller.abort();
    };

  }, [selectedPredictionId, authToken]);


  // ========================================================
  // 4. PREPARE WATERFALL DATA
  // ========================================================

  // Convert the contribution object into an array.
  const contributions = explanation
    ? Object.entries(explanation.feature_contributions)
        .filter(([, value]) => Number.isFinite(Number(value)))
        .map(([feature, value]) => ({
          feature,
          contribution: Number(value),
        }))
    : [];

  // Start the cumulative calculation at the SHAP baseline.
  let cumulativeValue = explanation
    ? explanation.base_value
    : 0;

  // Calculate the starting and ending positions of every bar.
  const waterfallSteps = contributions.map((item) => {

    // Store the model output before this feature contribution.
    const startValue = cumulativeValue;

    // Add the feature contribution to the cumulative output.
    cumulativeValue += item.contribution;

    // Store the model output after this feature contribution.
    const endValue = cumulativeValue;

    // Return the complete waterfall bar information.
    return {
      feature: item.feature,
      contribution: item.contribution,
      startValue,
      endValue,
    };
  });


  // ========================================================
  // 5. PREPARE GRAPH DIMENSIONS AND SCALE
  // ========================================================

  // Define the SVG graph width.
  const chartWidth = 1100;

  // Define the SVG graph height based on feature count.
  const chartHeight = Math.max(
    360,
    waterfallSteps.length * 58 + 150
  );

  // Reserve space for feature labels on the left.
  const plotLeft = 280;

  // Reserve space on the right for numerical labels.
  const plotRight = 1040;

  // Define the available horizontal graph width.
  const plotWidth = plotRight - plotLeft;

  // Collect all cumulative values for the graph scale.
  const allValues = explanation
    ? [
        explanation.base_value,
        ...waterfallSteps.map((item) => item.startValue),
        ...waterfallSteps.map((item) => item.endValue),
        explanation.model_output,
      ]
    : [0];

  // Find the smallest cumulative value.
  const minimumValue = Math.min(...allValues);

  // Find the largest cumulative value.
  const maximumValue = Math.max(...allValues);

  // Add some visual space around the graph values.
  const valuePadding =
    Math.max(maximumValue - minimumValue, 0.1) * 0.12;

  // Calculate the minimum visible graph value.
  const graphMinimum = minimumValue - valuePadding;

  // Calculate the maximum visible graph value.
  const graphMaximum = maximumValue + valuePadding;

  // Convert a model output value into an SVG X coordinate.
  const getX = (value) => {
    return plotLeft +
      ((value - graphMinimum) / (graphMaximum - graphMinimum)) *
      plotWidth;
  };


  // ========================================================
  // 6. LOADING AND ERROR STATES
  // ========================================================

  // Display a message while history is loading.
  if (historyLoading) {
    return (
      <section className="waterfall-panel">
        <p>Loading prediction history...</p>
      </section>
    );
  }

  // Display an error if history could not be retrieved.
  if (historyError) {
    return (
      <section className="waterfall-panel">
        <h3>SHAP Waterfall Explanation</h3>
        <p className="waterfall-error">{historyError}</p>
      </section>
    );
  }

  // Display a message if the user has no previous predictions.
  if (predictionHistory.length === 0) {
    return (
      <section className="waterfall-panel">
        <h3>SHAP Waterfall Explanation</h3>
        <p>No prediction history is available.</p>
        <p>Create a new prediction to view its explanation.</p>
      </section>
    );
  }


  // ========================================================
  // 7. MAIN WATERFALL INTERFACE
  // ========================================================

  return (

    <section className="waterfall-panel">

      {/* Main heading and description. */}
      <div className="waterfall-heading">

        <div>

          {/* Section identification label. */}
          <p className="section-eyebrow">
            INDIVIDUAL EXPLAINABLE AI
          </p>

          {/* Main visualization title. */}
          <h3>
            SHAP Waterfall Plot
          </h3>

          {/* Explain the purpose of this graph. */}
          <p>
            Understand how individual features contribute
            to a selected prediction.
          </p>

        </div>

        {/* Show the number of available predictions. */}
        <span className="waterfall-record-count">
          {predictionHistory.length} Predictions
        </span>

      </div>


      {/* ====================================================
          8. PREDICTION SELECTION
         ==================================================== */}

      {!lockSelection && (
        <div className="waterfall-selector">

          {/* Label for the prediction dropdown. */}
          <label htmlFor="waterfall-prediction">
            Select Prediction
          </label>

          {/* Dropdown containing the user's actual history. */}
          <select
            id="waterfall-prediction"
            value={selectedPredictionId}
            onChange={(event) => {
              setSelectedPredictionId(event.target.value);
            }}
          >

            {/* Display each prediction using its real ID. */}
            {predictionHistory.map((prediction) => (

              <option
                key={prediction.id}
                value={String(prediction.id)}
              >
                Prediction #{prediction.id}
                {' | '}
                Glucose: {prediction.glucose}
                {' | '}
                BMI: {prediction.bmi}
                {' | '}
                Age: {prediction.age}
              </option>

            ))}

          </select>

        </div>
      )}


      {/* ====================================================
          9. EXPLANATION LOADING AND ERROR
         ==================================================== */}

      {explanationLoading ? (

        <div className="waterfall-status">
          Loading individual SHAP explanation...
        </div>

      ) : explanationError ? (

        <p className="waterfall-error">
          {explanationError}
        </p>

      ) : explanation ? (

        <>

          {/* ==================================================
              10. BASE VALUE AND FINAL MODEL OUTPUT
             ================================================== */}

          <div className="waterfall-summary">

            {/* Display the SHAP baseline. */}
            <div className="waterfall-summary-card">

              <span>
                Base Value
              </span>

              <strong>
                {explanation.base_value.toFixed(4)}
              </strong>

            </div>

            {/* Display the final model output. */}
            <div className="waterfall-summary-card">

              <span>
                Final Model Output
              </span>

              <strong>
                {explanation.model_output.toFixed(4)}
              </strong>

            </div>

          </div>


          {/* ==================================================
              11. WATERFALL GRAPH
             ================================================== */}

          <div className="waterfall-chart-container">

            <svg
              className="waterfall-svg"
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              role="img"
              aria-label="SHAP waterfall chart showing individual feature contributions"
            >

              {/* Horizontal zero reference line. */}
              <line
                x1={plotLeft}
                y1="55"
                x2={plotRight}
                y2="55"
                stroke="#334155"
                strokeDasharray="5 5"
              />

              {/* Draw the baseline marker. */}
              <line
                x1={getX(explanation.base_value)}
                y1="75"
                x2={getX(explanation.base_value)}
                y2={chartHeight - 65}
                stroke="#a78bfa"
                strokeWidth="2"
                strokeDasharray="6 5"
              />

              {/* Draw one waterfall bar for every feature. */}
              {waterfallSteps.map((item, index) => {

                // Calculate the vertical position of this feature.
                const y = 100 + index * 58;

                // Convert the start value into an X coordinate.
                const startX = getX(item.startValue);

                // Convert the end value into an X coordinate.
                const endX = getX(item.endValue);

                // Calculate the bar's left position.
                const barX = Math.min(startX, endX);

                // Calculate the bar's width.
                const barWidth = Math.max(
                  Math.abs(endX - startX),
                  2
                );

                // Positive values increase the model output.
                const isPositive = item.contribution >= 0;

                // Choose a colour based on contribution direction.
                const barColor = isPositive
                  ? '#fb7185'
                  : '#38bdf8';

                return (

                  <g key={item.feature}>

                    {/* Display the feature name. */}
                    <text
                      x="15"
                      y={y + 17}
                      fill="#cbd5e1"
                      fontSize="15"
                    >
                      {item.feature}
                    </text>

                    {/* Display the feature contribution value. */}
                    <text
                      x={plotLeft - 12}
                      y={y + 17}
                      fill={barColor}
                      fontSize="13"
                      textAnchor="end"
                      fontWeight="bold"
                    >
                      {item.contribution > 0 ? '+' : ''}
                      {item.contribution.toFixed(4)}
                    </text>

                    {/* Draw the actual SHAP contribution bar. */}
                    <rect
                      x={barX}
                      y={y}
                      width={barWidth}
                      height="27"
                      rx="5"
                      fill={barColor}
                      opacity="0.9"
                    >

                      {/* Show detailed information on hover. */}
                      <title>
                        {item.feature}: {item.contribution.toFixed(6)}
                        {' | Start: '}
                        {item.startValue.toFixed(6)}
                        {' | End: '}
                        {item.endValue.toFixed(6)}
                      </title>

                    </rect>

                    {/* Connect this bar to the next cumulative value. */}
                    {index < waterfallSteps.length - 1 && (

                      <line
                        x1={endX}
                        y1={y + 27}
                        x2={endX}
                        y2={y + 58}
                        stroke="#64748b"
                        strokeWidth="1"
                        strokeDasharray="4 4"
                      />

                    )}

                  </g>

                );
              })}


              {/* ==================================================
                  12. FINAL OUTPUT MARKER
                 ================================================== */}

              {/* Draw the final model output marker. */}
              <line
                x1={getX(explanation.model_output)}
                y1="75"
                x2={getX(explanation.model_output)}
                y2={chartHeight - 65}
                stroke="#22d3ee"
                strokeWidth="2"
                strokeDasharray="6 5"
              />

              {/* Display the final output label. */}
              <text
                x={getX(explanation.model_output)}
                y={chartHeight - 35}
                textAnchor="middle"
                fill="#22d3ee"
                fontSize="14"
                fontWeight="bold"
              >
                Final: {explanation.model_output.toFixed(4)}
              </text>

              {/* Display the baseline label. */}
              <text
                x={getX(explanation.base_value)}
                y="35"
                textAnchor="middle"
                fill="#c4b5fd"
                fontSize="13"
              >
                Base: {explanation.base_value.toFixed(4)}
              </text>

            </svg>

          </div>


          {/* ==================================================
              13. GRAPH LEGEND
             ================================================== */}

          <div className="waterfall-legend">

            {/* Negative contribution legend. */}
            <span>
              <i className="waterfall-legend-negative" />
              Decreases model output
            </span>

            {/* Positive contribution legend. */}
            <span>
              <i className="waterfall-legend-positive" />
              Increases model output
            </span>

          </div>


          {/* ==================================================
              14. HOW TO INTERPRET
             ================================================== */}

          <div className="waterfall-note">

            <strong>
              How to interpret this graph
            </strong>

            <p>
              The explanation begins at the base value.
              Each feature contribution moves the model output
              upward or downward. Adding all contributions to
              the base value reconstructs the final model output.
            </p>

          </div>


          {/* ==================================================
              15. RESEARCH DISCLAIMER
             ================================================== */}

          <p className="waterfall-disclaimer">
            SHAP contributions explain the behaviour of this
            machine-learning model. They do not establish
            medical causation, diagnose diabetes, or replace
            professional medical evaluation. Model output is
            expressed in log-odds, not probability.
          </p>

        </>

      ) : null}

    </section>
  );
}


// ==========================================================
// 16. EXPORT COMPONENT
// ==========================================================

// Make the component available to App.jsx.
export default ShapWaterfall;
