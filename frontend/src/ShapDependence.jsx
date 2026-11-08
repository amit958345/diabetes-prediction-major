
/*
============================================================
DIAPREDICT - SHAP DEPENDENCE PLOT
============================================================

PURPOSE:
1. Fetch actual SHAP values from the FastAPI backend.
2. Display the relationship between feature values and SHAP values.
3. Allow users to select an input feature.
4. Allow users to select a colour interaction feature.
5. Display all dataset records as scatter plot points.
6. Handle loading, authentication and API errors.

DATA SOURCE:
GET /predictions/explanations/beeswarm

GRAPH:
X-axis = Original feature value.
Y-axis = SHAP contribution.
Dot colour = Selected interaction feature value.

IMPORTANT:
This graph explains model behaviour.
It does not establish medical causation.

============================================================
*/

// Import React hooks for state, API requests and calculations.
import { useEffect, useMemo, useState } from 'react';
import { API_BASE_URL } from './config';

// ==========================================================
// 1. SHAP DEPENDENCE COMPONENT
// ==========================================================

function ShapDependence({ authToken }) {

  // Store the actual beeswarm API response.
  const [shapData, setShapData] = useState(null);

  // Store the feature selected for the horizontal axis.
  const [selectedFeature, setSelectedFeature] = useState('Glucose');

  // Store the feature selected for dot colouring.
  const [interactionFeature, setInteractionFeature] = useState('BMI');

  // Track whether the API request is running.
  const [loading, setLoading] = useState(true);

  // Store API, authentication and validation errors.
  const [error, setError] = useState('');


  // ========================================================
  // 2. FETCH ACTUAL SHAP DATA FROM FASTAPI
  // ========================================================

  useEffect(() => {

    // Require a valid authentication token.
    if (!authToken) {
      setError('Please log in to view SHAP dependence analysis.');
      setLoading(false);
      return;
    }

    // Create a controller for cancelling unfinished requests.
    const controller = new AbortController();

    // Define the asynchronous API request.
    const fetchShapData = async () => {

      // Show the loading state.
      setLoading(true);

      // Clear previous errors.
      setError('');

      // Request actual SHAP values from the backend.
      try {

        // Send an authenticated GET request to FastAPI.
        const response = await fetch(
          `${API_BASE_URL}/predictions/explanations/beeswarm`,
          {
            // GET retrieves the existing SHAP dataset.
            method: 'GET',

            // Attach the JWT access token.
            headers: {
              'Authorization': `Bearer ${authToken}`,
              'Content-Type': 'application/json',
            },

            // Allow cancellation if the component is removed.
            signal: controller.signal,
          }
        );

        // Convert the response body into JavaScript data.
        const data = await response.json();

        // Handle unsuccessful HTTP responses.
        if (!response.ok) {
          throw new Error(
            data.detail || 'Unable to load SHAP dependence data.'
          );
        }

        // Validate the required API response fields.
        if (
          !Array.isArray(data.feature_names) ||
          !Array.isArray(data.feature_values) ||
          !Array.isArray(data.shap_values)
        ) {
          throw new Error('Invalid SHAP API response structure.');
        }

        // Ensure SHAP and original feature records match.
        if (
          data.feature_values.length !== data.shap_values.length
        ) {
          throw new Error('Feature and SHAP record counts do not match.');
        }

        // Store the actual backend response.
        setShapData(data);

      } catch (err) {

        // Ignore errors caused by intentional request cancellation.
        if (err.name === 'AbortError') {
          return;
        }

        // Display other errors.
        setError(err.message);

      } finally {

        // Stop loading if the request was not cancelled.
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };

    // Execute the API request.
    fetchShapData();

    // Cancel the request when the component is removed.
    return () => {
      controller.abort();
    };

  }, [authToken]);


  // ========================================================
  // 3. FIND SELECTED FEATURE INDEXES
  // ========================================================

  // Find the column index of the selected feature.
  const selectedFeatureIndex = shapData
    ? shapData.feature_names.indexOf(selectedFeature)
    : -1;

  // Find the column index of the selected interaction feature.
  const interactionFeatureIndex = shapData
    ? shapData.feature_names.indexOf(interactionFeature)
    : -1;


  // ========================================================
  // 4. PREPARE SCATTER PLOT DATA
  // ========================================================

  // Calculate the actual coordinates for every record.
  const chartData = useMemo(() => {

    // Return no chart data until the API response is available.
    if (
      !shapData ||
      selectedFeatureIndex < 0 ||
      interactionFeatureIndex < 0
    ) {
      return null;
    }

    // Define the SVG chart dimensions.
    const width = 1000;

    // Define the total graph height.
    const height = 520;

    // Reserve space for axis labels.
    const leftPadding = 100;
    const rightPadding = 45;
    const topPadding = 35;
    const bottomPadding = 85;

    // Calculate the usable plotting area.
    const plotWidth = width - leftPadding - rightPadding;
    const plotHeight = height - topPadding - bottomPadding;

    // Extract original selected-feature values.
    const xValues = shapData.feature_values
      .map((row) => row[selectedFeatureIndex])
      .filter(
        (value) =>
          value !== null &&
          Number.isFinite(Number(value))
      )
      .map(Number);

    // Extract selected-feature SHAP contributions.
    const yValues = shapData.shap_values
      .map((row) => row[selectedFeatureIndex])
      .filter((value) => Number.isFinite(Number(value)))
      .map(Number);

    // Extract the interaction feature's original values.
    const colourValues = shapData.feature_values
      .map((row) => row[interactionFeatureIndex])
      .filter(
        (value) =>
          value !== null &&
          Number.isFinite(Number(value))
      )
      .map(Number);

    // Stop if the selected feature has no valid data.
    if (
      xValues.length === 0 ||
      yValues.length === 0 ||
      colourValues.length === 0
    ) {
      return null;
    }

    // Find the minimum and maximum X-axis values.
    const xMinimum = Math.min(...xValues);
    const xMaximum = Math.max(...xValues);

    // Find the minimum and maximum SHAP values.
    const yMinimum = Math.min(...yValues);
    const yMaximum = Math.max(...yValues);

    // Add a small margin to the X-axis range.
    const xRange = Math.max(
      0.000001,
      xMaximum - xMinimum
    );

    // Add a small margin to the Y-axis range.
    const yRange = Math.max(
      0.000001,
      yMaximum - yMinimum
    );

    // Calculate padded axis limits.
    const xAxisMinimum = xMinimum - xRange * 0.05;
    const xAxisMaximum = xMaximum + xRange * 0.05;

    const yAxisMinimum = yMinimum - yRange * 0.1;
    const yAxisMaximum = yMaximum + yRange * 0.1;

    // Find the minimum and maximum interaction values.
    const colourMinimum = Math.min(...colourValues);
    const colourMaximum = Math.max(...colourValues);

    // Prepare one scatter point for each dataset record.
    const points = shapData.shap_values.map(
      (shapRow, recordIndex) => {

        // Retrieve the original selected-feature value.
        const originalValue =
          shapData.feature_values[recordIndex][selectedFeatureIndex];

        // Retrieve the selected feature's SHAP contribution.
        const shapValue = Number(
          shapRow[selectedFeatureIndex]
        );

        // Retrieve the original interaction-feature value.
        const colourValue =
          shapData.feature_values[recordIndex][interactionFeatureIndex];

        // Skip records with missing or invalid measurements.
        if (
          originalValue === null ||
          colourValue === null ||
          !Number.isFinite(Number(originalValue)) ||
          !Number.isFinite(shapValue) ||
          !Number.isFinite(Number(colourValue))
        ) {
          return null;
        }

        // Convert the original feature value into an X coordinate.
        const x =
          leftPadding +
          ((Number(originalValue) - xAxisMinimum) /
            (xAxisMaximum - xAxisMinimum)) *
            plotWidth;

        // Convert the SHAP contribution into a Y coordinate.
        // SVG Y coordinates increase downward, so the formula is reversed.
        const y =
          topPadding +
          ((yAxisMaximum - shapValue) /
            (yAxisMaximum - yAxisMinimum)) *
            plotHeight;

        // Calculate the interaction feature's colour intensity.
        const normalizedColour =
          colourMaximum > colourMinimum
            ? (Number(colourValue) - colourMinimum) /
              (colourMaximum - colourMinimum)
            : 0.5;

        // Return the complete scatter point.
        return {
          recordIndex,
          originalValue: Number(originalValue),
          shapValue,
          colourValue: Number(colourValue),
          normalizedColour,
          x,
          y,
        };
      }
    );

    // Remove records that contained missing values.
    const validPoints = points.filter(Boolean);

    // Calculate the horizontal position of the zero SHAP line.
    const zeroY =
      topPadding +
      ((yAxisMaximum - 0) /
        (yAxisMaximum - yAxisMinimum)) *
        plotHeight;

    // Return the completed chart data.
    return {
      width,
      height,
      leftPadding,
      topPadding,
      plotWidth,
      plotHeight,
      xAxisMinimum,
      xAxisMaximum,
      yAxisMinimum,
      yAxisMaximum,
      colourMinimum,
      colourMaximum,
      zeroY,
      points: validPoints,
    };

  }, [
    shapData,
    selectedFeatureIndex,
    interactionFeatureIndex,
  ]);


  // ========================================================
  // 5. COLOUR FUNCTION
  // ========================================================

  // Convert interaction feature values into blue-to-red colours.
  const getPointColor = (value) => {

    // Clamp the value between zero and one.
    const intensity = Math.max(
      0,
      Math.min(1, value)
    );

    // Define the blue colour for low feature values.
    const lowColor = [59, 130, 246];

    // Define the red colour for high feature values.
    const highColor = [239, 68, 68];

    // Interpolate the red channel.
    const red = Math.round(
      lowColor[0] +
      (highColor[0] - lowColor[0]) * intensity
    );

    // Interpolate the green channel.
    const green = Math.round(
      lowColor[1] +
      (highColor[1] - lowColor[1]) * intensity
    );

    // Interpolate the blue channel.
    const blue = Math.round(
      lowColor[2] +
      (highColor[2] - lowColor[2]) * intensity
    );

    // Return the final RGB colour.
    return `rgb(${red}, ${green}, ${blue})`;
  };


  // ========================================================
  // 6. LOADING STATE
  // ========================================================

  // Display a message while the backend request is running.
  if (loading) {
    return (
      <section className="dependence-panel">
        <p>Loading SHAP dependence analysis...</p>
      </section>
    );
  }


  // ========================================================
  // 7. ERROR STATE
  // ========================================================

  // Display an API or authentication error.
  if (error) {
    return (
      <section className="dependence-panel">

        <h3>SHAP Dependence Plot</h3>

        <p className="dependence-error">
          {error}
        </p>

      </section>
    );
  }


  // ========================================================
  // 8. GRAPH DATA VALIDATION
  // ========================================================

  // Display a message if no valid points are available.
  if (!chartData || chartData.points.length === 0) {
    return (
      <section className="dependence-panel">
        <p>No valid SHAP dependence data is available.</p>
      </section>
    );
  }


  // ========================================================
  // 9. DISPLAY THE DEPENDENCE PLOT
  // ========================================================

  return (

    <section className="dependence-panel">

      {/* Display the visualization heading. */}
      <div className="dependence-heading">

        <div>

          <p className="section-eyebrow">
            GLOBAL EXPLAINABLE AI
          </p>

          <h3>
            SHAP Dependence Plot
          </h3>

          <p>
            Explore how feature values relate to their SHAP contributions.
          </p>

        </div>

        {/* Display the number of valid records. */}
        <span className="dependence-record-count">
          {chartData.points.length} Records
        </span>

      </div>


      {/* Feature selection controls. */}
      <div className="dependence-controls">

        {/* Select the feature plotted on the X-axis. */}
        <div className="dependence-control-group">

          <label htmlFor="dependence-feature">
            Feature (X-axis)
          </label>

          <select
            id="dependence-feature"
            value={selectedFeature}
            onChange={(event) =>
              setSelectedFeature(event.target.value)
            }
          >

            {shapData.feature_names.map((feature) => (

              <option
                key={feature}
                value={feature}
              >
                {feature}
              </option>

            ))}

          </select>

        </div>


        {/* Select the feature used to colour the dots. */}
        <div className="dependence-control-group">

          <label htmlFor="dependence-interaction">
            Colour by feature
          </label>

          <select
            id="dependence-interaction"
            value={interactionFeature}
            onChange={(event) =>
              setInteractionFeature(event.target.value)
            }
          >

            {shapData.feature_names.map((feature) => (

              <option
                key={feature}
                value={feature}
              >
                {feature}
              </option>

            ))}

          </select>

        </div>

      </div>


      {/* Display the feature-value colour legend. */}
      <div className="dependence-legend">

        <span>Low</span>

        <div className="dependence-color-gradient" />

        <span>High</span>

        <span className="dependence-legend-feature">
          {interactionFeature}
        </span>

      </div>


      {/* Allow horizontal scrolling on smaller screens. */}
      <div className="dependence-chart-container">

        <svg
          className="dependence-svg"
          viewBox={`0 0 ${chartData.width} ${chartData.height}`}
          role="img"
          aria-label={`${selectedFeature} SHAP dependence scatter plot`}
        >

          {/* Draw the horizontal and vertical graph axes. */}
          <line
            x1={chartData.leftPadding}
            y1={chartData.topPadding}
            x2={chartData.leftPadding}
            y2={chartData.topPadding + chartData.plotHeight}
            stroke="#64748b"
            strokeWidth="1.5"
          />

          <line
            x1={chartData.leftPadding}
            y1={chartData.topPadding + chartData.plotHeight}
            x2={chartData.leftPadding + chartData.plotWidth}
            y2={chartData.topPadding + chartData.plotHeight}
            stroke="#64748b"
            strokeWidth="1.5"
          />


          {/* Draw the zero SHAP reference line. */}
          {chartData.zeroY >= chartData.topPadding &&
            chartData.zeroY <=
              chartData.topPadding + chartData.plotHeight && (

              <g>

                <line
                  x1={chartData.leftPadding}
                  y1={chartData.zeroY}
                  x2={chartData.leftPadding + chartData.plotWidth}
                  y2={chartData.zeroY}
                  stroke="#94a3b8"
                  strokeWidth="1.5"
                  strokeDasharray="7 5"
                />

                <text
                  x={chartData.leftPadding + 8}
                  y={chartData.zeroY - 8}
                  fill="#94a3b8"
                  fontSize="12"
                >
                  Zero SHAP
                </text>

              </g>

            )}


          {/* Draw five horizontal grid lines and Y-axis labels. */}
          {[0, 1, 2, 3, 4].map((index) => {

            // Calculate the fraction along the Y-axis.
            const fraction = index / 4;

            // Calculate the Y coordinate.
            const y =
              chartData.topPadding +
              fraction * chartData.plotHeight;

            // Calculate the corresponding SHAP value.
            const value =
              chartData.yAxisMaximum -
              fraction *
                (chartData.yAxisMaximum - chartData.yAxisMinimum);

            return (
              <g key={index}>

                {/* Draw the horizontal grid line. */}
                <line
                  x1={chartData.leftPadding}
                  y1={y}
                  x2={chartData.leftPadding + chartData.plotWidth}
                  y2={y}
                  stroke="#263b55"
                  strokeDasharray="4 5"
                />

                {/* Display the SHAP value. */}
                <text
                  x={chartData.leftPadding - 12}
                  y={y + 4}
                  textAnchor="end"
                  fill="#a9bfd8"
                  fontSize="12"
                >
                  {value.toFixed(2)}
                </text>

              </g>
            );
          })}


          {/* Draw five vertical grid lines and X-axis labels. */}
          {[0, 1, 2, 3, 4].map((index) => {

            // Calculate the fraction along the X-axis.
            const fraction = index / 4;

            // Calculate the X coordinate.
            const x =
              chartData.leftPadding +
              fraction * chartData.plotWidth;

            // Calculate the original feature value.
            const value =
              chartData.xAxisMinimum +
              fraction *
                (chartData.xAxisMaximum - chartData.xAxisMinimum);

            return (
              <g key={index}>

                {/* Draw the vertical grid line. */}
                <line
                  x1={x}
                  y1={chartData.topPadding}
                  x2={x}
                  y2={chartData.topPadding + chartData.plotHeight}
                  stroke="#263b55"
                  strokeDasharray="4 5"
                />

                {/* Display the feature value. */}
                <text
                  x={x}
                  y={chartData.topPadding + chartData.plotHeight + 22}
                  textAnchor="middle"
                  fill="#a9bfd8"
                  fontSize="12"
                >
                  {value.toFixed(1)}
                </text>

              </g>
            );
          })}


          {/* Draw all actual dataset records as scatter points. */}
          {chartData.points.map((point) => (

            <circle
              key={point.recordIndex}
              cx={point.x}
              cy={point.y}
              r="4"
              fill={getPointColor(point.normalizedColour)}
              fillOpacity="0.78"
              stroke="#0b1422"
              strokeWidth="0.4"
            >

              {/* Show the original record values when hovering. */}
              <title>
                {`Record: ${point.recordIndex + 1}
Feature: ${selectedFeature}
Feature value: ${point.originalValue.toFixed(3)}
SHAP contribution: ${point.shapValue.toFixed(5)}
Colour feature: ${interactionFeature}
Colour feature value: ${point.colourValue.toFixed(3)}`}
              </title>

            </circle>

          ))}


          {/* Display the horizontal axis title. */}
          <text
            x={chartData.leftPadding + chartData.plotWidth / 2}
            y={chartData.height - 20}
            textAnchor="middle"
            fill="#d9e8f7"
            fontSize="14"
          >
            {selectedFeature} value
          </text>


          {/* Display the vertical axis title. */}
          <text
            transform={`translate(25 ${
              chartData.topPadding + chartData.plotHeight / 2
            }) rotate(-90)`}
            textAnchor="middle"
            fill="#d9e8f7"
            fontSize="14"
          >
            SHAP value (log-odds)
          </text>

        </svg>

      </div>


      {/* Explain how to read the dependence plot. */}
      <div className="dependence-note">

        <strong>How to interpret:</strong>

        <p>
          Each dot represents one dataset record. The X-axis
          displays the original feature value, while the Y-axis
          displays its SHAP contribution. Positive SHAP values
          increase the model output, and negative values
          decrease it. Dot colour represents the selected
          interaction feature value.
        </p>

      </div>


      {/* Display the dataset limitation. */}
      <p className="dependence-disclaimer">
        Data source: Training dataset.
        This graph explains the trained model's behaviour.
        It does not establish medical causation or clinical validity.
      </p>

    </section>
  );
}


// Export the component for use in App.jsx.
export default ShapDependence;
