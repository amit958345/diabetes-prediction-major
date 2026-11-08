
/*
============================================================
DIAPREDICT - GLOBAL SHAP FEATURE IMPORTANCE
============================================================

PURPOSE:
1. Retrieve global SHAP feature importance from FastAPI.
2. Authenticate the request using the user's JWT token.
3. Sort features from highest to lowest importance.
4. Display actual backend values in a horizontal bar chart.
5. Handle loading and API errors.

IMPORTANT:
The graph uses real API data.
It does not contain manually created chart values.
============================================================
*/

import { useEffect, useState } from 'react';
import { API_BASE_URL } from './config';

// ==========================================================
// 1. GLOBAL SHAP COMPONENT
// ==========================================================

function GlobalShapImportance({ authToken }) {

  // Stores the global SHAP response from FastAPI.
  const [shapData, setShapData] = useState(null);

  // Tracks whether the API request is running.
  const [loading, setLoading] = useState(true);

  // Stores any API or authentication error.
  const [error, setError] = useState('');


  // ========================================================
  // 2. FETCH GLOBAL SHAP DATA
  // ========================================================

  useEffect(() => {

    // Stop if the user is not authenticated.
    if (!authToken) {
      setError('Please log in to view global SHAP analysis.');
      setLoading(false);
      return;
    }

    // Allows the request to be cancelled if the component
    // is removed before the API response arrives.
    const controller = new AbortController();


    // Retrieves global feature importance from FastAPI.
    const fetchGlobalShap = async () => {

      setLoading(true);
      setError('');

      try {

        // Send an authenticated GET request.
        const response = await fetch(
          `${API_BASE_URL}/predictions/explanations/global`,
          {
            method: 'GET',

            headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json',
            },

            signal: controller.signal,
          }
        );

        // Convert the response body into JavaScript data.
        const data = await response.json();

        // Handle authentication and server errors.
        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load global SHAP data.'
          );
        }

        // Validate the expected API response structure.
        if (
          !data.feature_importance ||
          typeof data.feature_importance !== 'object'
        ) {
          throw new Error('Invalid SHAP response structure.');
        }

        // Store the real backend response.
        setShapData(data);

      } catch (err) {

        // Ignore errors caused by intentional cancellation.
        if (err.name === 'AbortError') {
          return;
        }

        // Display the API error.
        setError(err.message);

      } finally {

        // Stop loading only if the request was not cancelled.
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };


    // Execute the API request.
    fetchGlobalShap();


    // Cancel the request when the component is removed
    // or when the authentication token changes.
    return () => {
      controller.abort();
    };

  }, [authToken]);


  // ========================================================
  // 3. PREPARE GRAPH DATA
  // ========================================================

  // Convert the API object into an array.
  // Sort the features from highest to lowest importance.
  const sortedFeatures = shapData
    ? Object.entries(shapData.feature_importance)
        .filter(([, value]) => Number.isFinite(Number(value)))
        .map(([feature, value]) => ({
          feature,
          importance: Number(value),
        }))
        .sort((a, b) => b.importance - a.importance)
    : [];


  // Find the highest importance value.
  // This value is used to calculate relative bar widths.
  const maximumImportance = Math.max(
    0,
    ...sortedFeatures.map((item) => item.importance)
  );


  // ========================================================
  // 4. LOADING STATE
  // ========================================================

  if (loading) {
    return (
      <section className="global-shap-panel">
        <p>Loading global SHAP analysis...</p>
      </section>
    );
  }


  // ========================================================
  // 5. ERROR STATE
  // ========================================================

  if (error) {
    return (
      <section className="global-shap-panel">
        <h3>Global SHAP Analysis</h3>
        <p className="global-shap-error">{error}</p>
      </section>
    );
  }


  // ========================================================
  // 6. GRAPH DISPLAY
  // ========================================================

  return (

    <section className="global-shap-panel">

      {/* Graph heading */}
      <div className="global-shap-heading">

        <div>
          <p className="section-eyebrow">
            GLOBAL EXPLAINABLE AI
          </p>

          <h3>
            SHAP Feature Importance
          </h3>

          <p>
            Average absolute contribution of each feature
            across the analysed dataset.
          </p>
        </div>

        <span className="global-shap-record-count">
          {shapData.total_records_analysed} Records
        </span>

      </div>


      {/* Horizontal bar chart */}
      <div className="global-shap-chart">

        {sortedFeatures.map((item, index) => {

          // Calculate the bar width relative to the
          // highest feature importance value.
          const barWidth = maximumImportance > 0
            ? (item.importance / maximumImportance) * 100
            : 0;

          return (

            <div
              className="global-shap-row"
              key={item.feature}
            >

              {/* Feature name and numerical value */}
              <div className="global-shap-label">

                <span>
                  <strong>{index + 1}.</strong> {item.feature}
                </span>

                <strong>
                  {item.importance.toFixed(4)}
                </strong>

              </div>


              {/* Background track and importance bar */}
              <div className="global-shap-track">

                <div
                  className="global-shap-bar"
                  style={{
                    width: `${barWidth}%`,
                  }}
                />

              </div>

            </div>

          );
        })}

      </div>


      {/* Explain the meaning of the graph */}
      <div className="global-shap-note">

        <strong>How to interpret:</strong>

        <p>
          Higher mean absolute SHAP values indicate greater
          average influence on the model output. These values
          do not indicate whether a feature increases or
          decreases an individual prediction.
        </p>

      </div>


      {/* Research limitation */}
      <p className="global-shap-disclaimer">
        Global importance describes model behaviour on the
        analysed dataset. It does not establish medical
        causation or clinical significance.
      </p>

    </section>
  );
}


// Export the component for use in App.jsx.
export default GlobalShapImportance;
