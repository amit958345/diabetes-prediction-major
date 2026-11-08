
/*
============================================================
DIAPREDICT - SHAP BEESWARM VISUALIZATION
============================================================

PURPOSE:
1. Fetch actual per-record SHAP values from FastAPI.
2. Display the SHAP contribution of all eight features.
3. Display individual records as dots.
4. Colour dots according to original feature values.
5. Sort features using mean absolute SHAP importance.
6. Display tooltips for individual data points.

DATA SOURCE:
GET /predictions/explanations/beeswarm

IMPORTANT:
- Uses actual backend SHAP data.
- Does not generate fake prediction values.
- SHAP values represent model contributions in log-odds.
- This analysis uses the training dataset.

============================================================
*/

// Import React hooks for state, side effects and calculations.
import { useEffect, useMemo, useState } from 'react';
import { API_BASE_URL } from './config';

// ==========================================================
// 1. BEESWARM COMPONENT
// ==========================================================

function ShapBeeswarm({ authToken, onLogout }) {

  // Stores the beeswarm response received from FastAPI.
  const [beeswarmData, setBeeswarmData] = useState(null);

  // Tracks whether the backend request is running.
  const [loading, setLoading] = useState(true);

  // Stores authentication, network and validation errors.
  const [error, setError] = useState('');


  // ========================================================
  // 2. FETCH REAL BEESWARM DATA
  // ========================================================

  useEffect(() => {

    // Do not request protected data without a login token.
    if (!authToken) {
      setError('Please log in to view SHAP beeswarm analysis.');
      setLoading(false);
      return;
    }

    // Create a controller to cancel an unfinished request.
    const controller = new AbortController();

    // Define the asynchronous API request.
    const fetchBeeswarmData = async () => {

      // Start the loading indicator.
      setLoading(true);

      // Clear any previous error.
      setError('');

      // Clear old chart data while refreshing.
      setBeeswarmData(null);

      try {

        // Request actual per-record SHAP values from FastAPI.
        const response = await fetch(
          `${API_BASE_URL}/predictions/explanations/beeswarm`,
          {
            // Use GET because we are retrieving data.
            method: 'GET',

            // Send the JWT token for authentication.
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json',
            },

            // Allow the request to be cancelled.
            signal: controller.signal,
          }
        );

        // Automatically log out when the JWT token has expired.
        // Handle an expired or invalid login session.
        if (response.status === 401) {
          onLogout?.();

          throw new Error(
            'Your session has expired. Please log in again.'
          );
        }

        // Convert the JSON response into JavaScript data.
        const data = await response.json();


        // Handle HTTP errors returned by FastAPI.
        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load SHAP beeswarm data.'
          );
        }

        // Check that all required response fields exist.
        if (
          !Array.isArray(data.feature_names) ||
          !Array.isArray(data.shap_values) ||
          !Array.isArray(data.feature_values) ||
          typeof data.feature_importance !== 'object' ||
          data.feature_importance === null
        ) {
          throw new Error('Invalid beeswarm API response structure.');
        }

        // Check that both matrices contain the same number of records.
        if (
          data.shap_values.length !== data.feature_values.length
        ) {
          throw new Error('SHAP and feature-value record counts do not match.');
        }

        // Check that the API returned records.
        if (data.shap_values.length === 0) {
          throw new Error('The beeswarm dataset contains no records.');
        }

        // Check that each record contains eight feature values.
        const featureCount = data.feature_names.length;

        const invalidShapRow = data.shap_values.some(
          (row) =>
            !Array.isArray(row) ||
            row.length !== featureCount ||
            !row.every((value) => Number.isFinite(Number(value)))
        );

        // Validate the original feature-value matrix.
        const invalidFeatureRow = data.feature_values.some(
          (row) =>
            !Array.isArray(row) ||
            row.length !== featureCount ||
            !row.every(
              (value) =>
                value === null ||
                Number.isFinite(Number(value))
            )
        );

        // Stop if any returned row is invalid.
        if (invalidShapRow || invalidFeatureRow) {
          throw new Error('Invalid values found in the beeswarm dataset.');
        }

        // Store the actual backend response.
        setBeeswarmData(data);

      } catch (err) {

        // Ignore errors caused by intentional cancellation.
        if (err.name === 'AbortError') {
          return;
        }

        // Store the error message for display.
        setError(err.message);

      } finally {

        // Stop loading if the request was not cancelled.
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };

    // Execute the API request.
    fetchBeeswarmData();

    // Cancel the request when the component is removed.
    return () => {
      controller.abort();
    };

  }, [authToken]);


  // ========================================================
  // 3. PREPARE FEATURE IMPORTANCE ORDER
  // ========================================================

  // Sort the features from highest to lowest importance.
  // The API already returns mean absolute SHAP importance.
  const sortedFeatures = useMemo(() => {

    // Return an empty list before API data is available.
    if (!beeswarmData) {
      return [];
    }

    // Convert the importance object into an array.
    return Object.entries(beeswarmData.feature_importance)

      // Keep only valid numerical importance values.
      .filter(([, value]) => Number.isFinite(Number(value)))

      // Convert importance values into JavaScript numbers.
      .map(([feature, importance]) => ({
        feature,
        importance: Number(importance),
      }))

      // Sort from highest to lowest importance.
      .sort((a, b) => b.importance - a.importance);

  }, [beeswarmData]);


  // ========================================================
  // 4. PREPARE BEESWARM GRAPH COORDINATES
  // ========================================================

  // Calculate the complete graph layout using actual SHAP values.
  const chartData = useMemo(() => {

    // Do not calculate coordinates before data arrives.
    if (!beeswarmData || sortedFeatures.length === 0) {
      return null;
    }

    // Define the SVG graph dimensions.
    const chartWidth = 1000;

    // Reserve space for feature names on the left.
    const plotStart = 235;

    // Reserve space for the horizontal SHAP axis.
    const plotWidth = 725;

    // Define vertical spacing between feature rows.
    const rowHeight = 72;

    // Reserve space above and below the graph.
    const topPadding = 55;
    const bottomPadding = 65;

    // Calculate the full graph height.
    const chartHeight =
      topPadding +
      sortedFeatures.length * rowHeight +
      bottomPadding;

    // Find the largest absolute SHAP contribution.
    const allShapValues = beeswarmData.shap_values.flat();

    const maximumAbsoluteShap = Math.max(
      0.000001,
      ...allShapValues.map((value) => Math.abs(Number(value)))
    );

    // Add a small margin to both ends of the axis.
    const axisMaximum = maximumAbsoluteShap * 1.08;

    // Convert a SHAP value into its horizontal SVG coordinate.
    const getXPosition = (shapValue) => {

      // Map the negative-to-positive range onto the graph width.
      return plotStart +
        ((Number(shapValue) + axisMaximum) / (2 * axisMaximum)) *
        plotWidth;
    };

    // Prepare the graph data for every feature.
    const featureRows = sortedFeatures.map((feature, featureIndex) => {

      // Find the original feature index in the API response.
      const originalIndex = beeswarmData.feature_names.indexOf(
        feature.feature
      );

      // Get the original measurement for this feature across records.
      const featureValues = beeswarmData.feature_values
        .map((row) => row[originalIndex])
        .filter((value) => value !== null)
        .map(Number);

      // Calculate the minimum original feature value.
      const minimumValue = featureValues.length
        ? Math.min(...featureValues)
        : 0;

      // Calculate the maximum original feature value.
      const maximumValue = featureValues.length
        ? Math.max(...featureValues)
        : 0;

      // Calculate the vertical centre of this feature row.
      const rowCenter =
        topPadding + featureIndex * rowHeight + rowHeight / 2;

      // Pair each SHAP contribution with its original measurement.
      const points = beeswarmData.shap_values.map(
        (record, recordIndex) => {

          // Extract the SHAP contribution for this feature.
          const shapValue = Number(record[originalIndex]);

          // Extract the original feature measurement.
          const featureValue =
            beeswarmData.feature_values[recordIndex][originalIndex];

          // Calculate the feature-value colour intensity.
          let normalizedValue = 0.5;

          if (
            featureValue !== null &&
            maximumValue > minimumValue
          ) {
            normalizedValue =
              (Number(featureValue) - minimumValue) /
              (maximumValue - minimumValue);
          }

          // Return the data needed to draw this dot.
          return {
            recordIndex,
            shapValue,
            featureValue,
            normalizedValue,
            x: getXPosition(shapValue),
          };
        }
      );

      // Sort dots by horizontal SHAP contribution.
      points.sort((a, b) => a.x - b.x);

      // Track how many dots occupy each horizontal bin.
      const binCounts = {};

      // Calculate the vertical position of every dot.
      points.forEach((point) => {

        // Group nearby dots into small horizontal bins.
        const bin = Math.floor(point.x / 7);

        // Get the current number of dots in this bin.
        const count = binCounts[bin] || 0;

        // Increase the count for the next dot.
        binCounts[bin] = count + 1;

        // Arrange dots above and below the row centre.
        const lanePattern = [
          0, 1, -1, 2, -2, 3, -3, 4, -4
        ];

        // Select the vertical lane for this dot.
        const lane = lanePattern[
          count % lanePattern.length
        ];

        // Set the vertical SVG coordinate.
        point.y = rowCenter + lane * 6.2;
      });

      // Return the completed feature row.
      return {
        feature: feature.feature,
        importance: feature.importance,
        rowCenter,
        points,
      };
    });

    // Return the complete graph layout.
    return {
      chartWidth,
      chartHeight,
      plotStart,
      plotWidth,
      topPadding,
      bottomPadding,
      axisMaximum,
      featureRows,
      getXPosition,
    };

  }, [beeswarmData, sortedFeatures]);


  // ========================================================
  // 5. CONVERT FEATURE VALUES INTO DOT COLOURS
  // ========================================================

  // Use blue for low values and red for high values.
  const getPointColor = (normalizedValue) => {

    // Keep missing feature values neutral grey.
    if (normalizedValue === null) {
      return '#94a3b8';
    }

    // Clamp the colour intensity between zero and one.
    const intensity = Math.max(
      0,
      Math.min(1, normalizedValue)
    );

    // Define the low-value blue colour.
    const lowColor = [59, 130, 246];

    // Define the high-value red colour.
    const highColor = [239, 68, 68];

    // Interpolate between blue and red.
    const red = Math.round(
      lowColor[0] +
      (highColor[0] - lowColor[0]) * intensity
    );

    const green = Math.round(
      lowColor[1] +
      (highColor[1] - lowColor[1]) * intensity
    );

    const blue = Math.round(
      lowColor[2] +
      (highColor[2] - lowColor[2]) * intensity
    );

    // Return a valid RGB colour.
    return `rgb(${red}, ${green}, ${blue})`;
  };


  // ========================================================
  // 6. LOADING STATE
  // ========================================================

  // Display a loading message while waiting for FastAPI.
  if (loading) {
    return (
      <section className="beeswarm-panel">
        <p>Loading SHAP beeswarm analysis...</p>
      </section>
    );
  }


  // ========================================================
  // 7. ERROR STATE
  // ========================================================

  // Display an error message if the request fails.
  if (error) {
    return (
      <section className="beeswarm-panel">

        <h3>SHAP Beeswarm Analysis</h3>

        <p className="beeswarm-error">
          {error}
        </p>

      </section>
    );
  }


  // ========================================================
  // 8. EMPTY DATA STATE
  // ========================================================

  // Handle the situation where no chart data is available.
  if (!chartData) {
    return (
      <section className="beeswarm-panel">
        <p>No SHAP beeswarm data is available.</p>
      </section>
    );
  }


  // ========================================================
  // 9. RENDER THE BEESWARM VISUALIZATION
  // ========================================================

  return (

    <section className="beeswarm-panel">

      {/* Main visualization heading */}
      <div className="beeswarm-heading">

        <div>

          <p className="section-eyebrow">
            GLOBAL EXPLAINABLE AI
          </p>

          <h3>
            SHAP Beeswarm Plot
          </h3>

          <p>
            Individual feature contributions across the analysed dataset.
          </p>

        </div>

        {/* Display the number of analysed records */}
        <span className="beeswarm-record-count">
          {beeswarmData.total_records_analysed} Records
        </span>

      </div>


      {/* Colour legend */}
      <div className="beeswarm-legend">

        <span>Feature value:</span>

        <span className="beeswarm-legend-low">
          Low
        </span>

        <div className="beeswarm-color-gradient" />

        <span className="beeswarm-legend-high">
          High
        </span>

      </div>


      {/* Scrollable graph container for smaller screens */}
      <div className="beeswarm-chart-container">

        <svg
          className="beeswarm-svg"
          viewBox={`0 0 ${chartData.chartWidth} ${chartData.chartHeight}`}
          role="img"
          aria-label="SHAP beeswarm plot showing feature contributions"
        >

          {/* Draw vertical reference lines */}
          {[-1, -0.5, 0, 0.5, 1].map((tick) => {

            // Calculate the SHAP value represented by this tick.
            const value = tick * chartData.axisMaximum;

            // Convert the SHAP value into an SVG coordinate.
            const x = chartData.getXPosition(value);

            return (
              <g key={tick}>

                {/* Draw the vertical grid line */}
                <line
                  x1={x}
                  y1={chartData.topPadding - 15}
                  x2={x}
                  y2={
                    chartData.chartHeight -
                    chartData.bottomPadding
                  }
                  stroke={tick === 0 ? '#94a3b8' : '#263b55'}
                  strokeWidth={tick === 0 ? 2 : 1}
                  strokeDasharray={tick === 0 ? '0' : '4 5'}
                />

                {/* Display the numerical axis label */}
                <text
                  x={x}
                  y={chartData.chartHeight - 28}
                  fill="#a9bfd8"
                  fontSize="12"
                  textAnchor="middle"
                >
                  {value.toFixed(2)}
                </text>

              </g>
            );
          })}


          {/* Display each feature row */}
          {chartData.featureRows.map((row, index) => (

            <g key={row.feature}>

              {/* Display the feature ranking and name */}
              <text
                x="12"
                y={row.rowCenter + 4}
                fill="#d9e8f7"
                fontSize="13"
              >
                {index + 1}. {row.feature}
              </text>

              {/* Draw a subtle separator between feature rows */}
              <line
                x1={chartData.plotStart}
                y1={row.rowCenter + 32}
                x2={chartData.plotStart + chartData.plotWidth}
                y2={row.rowCenter + 32}
                stroke="#1d3047"
                strokeWidth="1"
              />

              {/* Draw the individual record dots */}
              {row.points.map((point) => (

                <circle
                  key={point.recordIndex}
                  cx={point.x}
                  cy={point.y}
                  r="3.2"
                  fill={getPointColor(
                    point.featureValue === null
                      ? null
                      : point.normalizedValue
                  )}
                  fillOpacity="0.78"
                  stroke="#0b1422"
                  strokeWidth="0.35"
                >

                  {/* Display actual record information on hover */}
                  <title>
                    {`Feature: ${row.feature}
Record: ${point.recordIndex + 1}
SHAP value: ${point.shapValue.toFixed(5)}
Feature value: ${
  point.featureValue === null
    ? 'Missing'
    : point.featureValue
}`}
                  </title>

                </circle>

              ))}

            </g>

          ))}


          {/* Label the horizontal SHAP axis */}
          <text
            x={chartData.plotStart + chartData.plotWidth / 2}
            y={chartData.chartHeight - 5}
            fill="#d9e8f7"
            fontSize="13"
            textAnchor="middle"
          >
            SHAP value (model output in log-odds)
          </text>

        </svg>

      </div>


      {/* Explain how to interpret the visualization */}
      <div className="beeswarm-note">

        <strong>How to interpret:</strong>

        <p>
          Each dot represents one record. Dots to the right
          have positive SHAP contributions and increase the
          model output. Dots to the left have negative
          contributions and decrease the model output.
          Blue represents lower feature values, while red
          represents higher feature values.
        </p>

      </div>


      {/* Display the dataset limitation */}
      <p className="beeswarm-disclaimer">
        Data source: {beeswarmData.data_source}.
        This visualization explains model behaviour on the
        analysed dataset. It does not establish causation,
        medical risk or clinical validity.
      </p>

    </section>
  );
}


// Export the component so App.jsx can use it.
export default ShapBeeswarm;
