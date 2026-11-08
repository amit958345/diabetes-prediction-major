/*
============================================================
DIAPREDICT - AI-POWERED DIABETES RISK PREDICTION
============================================================

PURPOSE:
1. Collect eight patient health-related features.
2. Validate the entered information.
3. Send patient information to the FastAPI backend.
4. Authenticate the request using the user's JWT token.
5. Display the actual Logistic Regression prediction.
6. Display class probabilities.
7. Display individual SHAP feature explanations.

BACKEND:
POST ${API_BASE_URL}/predictions/predict

IMPORTANT:
This is an academic research application.
It does not provide a clinical diagnosis.
============================================================
*/

import { useState } from 'react'
import { API_BASE_URL } from './config'

// ============================================================
// 1. INPUT FIELD CONFIGURATION
// ============================================================

// Defines the eight features expected by the trained ML model.
// The names here match the existing React form state.
// The API payload will be converted to snake_case separately.

const inputFields = [
  {
    name: 'Pregnancies',
    label: 'Number of Pregnancies',
    unit: 'Count',
    min: 0,
    max: 20,
    step: 1,
    placeholder: 'e.g. 2',
  },
  {
    name: 'Glucose',
    label: 'Glucose',
    unit: 'mg/dL',
    min: 0.1,
    max: 300,
    step: 0.1,
    placeholder: 'e.g. 120',
  },
  {
    name: 'BloodPressure',
    label: 'Blood Pressure',
    unit: 'mm Hg',
    min: 0.1,
    max: 200,
    step: 0.1,
    placeholder: 'e.g. 70',
  },
  {
    name: 'SkinThickness',
    label: 'Skin Thickness',
    unit: 'mm',
    min: 0.1,
    max: 100,
    step: 0.1,
    placeholder: 'e.g. 25',
  },
  {
    name: 'Insulin',
    label: 'Insulin',
    unit: 'μU/mL',
    min: 0.1,
    max: 1000,
    step: 0.1,
    placeholder: 'e.g. 80',
  },
  {
    name: 'BMI',
    label: 'Body Mass Index (BMI)',
    unit: 'kg/m²',
    min: 0.1,
    max: 100,
    step: 0.1,
    placeholder: 'e.g. 30.5',
  },
  {
    name: 'DiabetesPedigreeFunction',
    label: 'Diabetes Pedigree Function',
    unit: 'Score',
    min: 0,
    max: 3,
    step: 0.001,
    placeholder: 'e.g. 0.45',
  },
  {
    name: 'Age',
    label: 'Age',
    unit: 'Years',
    min: 1,
    max: 120,
    step: 1,
    placeholder: 'e.g. 35',
  },
]


// ============================================================
// 2. INITIAL FORM DATA
// ============================================================

// Keeps all patient input fields empty when the form opens.
// Each property corresponds to one ML model feature.

const initialFormData = {
  Pregnancies: '',
  Glucose: '',
  BloodPressure: '',
  SkinThickness: '',
  Insulin: '',
  BMI: '',
  DiabetesPedigreeFunction: '',
  Age: '',
}


// ============================================================
// 3. PREDICTION COMPONENT
// ============================================================

// authToken is received from App.jsx.
// It is the JWT access token returned by the login API.

function Prediction({ authToken, onLogout }) {

  // Stores the values entered in the eight form fields.
  const [formData, setFormData] = useState(initialFormData)

  // Stores validation messages and API error messages.
  const [message, setMessage] = useState('')

  // Stores the actual prediction response returned by FastAPI.
  const [predictionResult, setPredictionResult] = useState(null)

  // Tracks whether the backend request is currently running.
  const [loading, setLoading] = useState(false)


  // ==========================================================
  // 4. HANDLE INPUT CHANGES
  // ==========================================================

  function handleInputChange(event) {

    // Extract the changed input's name and value.
    const { name, value } = event.target

    // Update only the field that the user has changed.
    setFormData((previousData) => ({
      ...previousData,
      [name]: value,
    }))

    // Remove the previous result when new patient information
    // is entered, so an old result is not mistaken for a new one.
    setMessage('')
    setPredictionResult(null)
  }


  // ==========================================================
  // 5. CONVERT FORM DATA INTO BACKEND FORMAT
  // ==========================================================

  function prepareApiPayload(numericData) {

    // The backend PredictionInput schema expects snake_case.
    // React form names are converted into the exact API names.

    return {
      pregnancies: numericData.Pregnancies,
      glucose: numericData.Glucose,
      blood_pressure: numericData.BloodPressure,
      skin_thickness: numericData.SkinThickness,
      insulin: numericData.Insulin,
      bmi: numericData.BMI,
      diabetes_pedigree_function:
        numericData.DiabetesPedigreeFunction,
      age: numericData.Age,
    }
  }


  // ==========================================================
  // 6. HANDLE FORM SUBMISSION
  // ==========================================================

  async function handleSubmit(event) {

    // Prevent the browser's default form submission and refresh.
    event.preventDefault()

    // Clear any previous feedback and prediction.
    setMessage('')
    setPredictionResult(null)


    // --------------------------------------------------------
    // STEP 1: VERIFY AUTHENTICATION
    // --------------------------------------------------------

    // The prediction API requires a valid JWT access token.
    if (!authToken) {
      setMessage('Your login session is missing. Please log in again.')
      return
    }


    // --------------------------------------------------------
    // STEP 2: CHECK FOR EMPTY FIELDS
    // --------------------------------------------------------

    const hasEmptyField = Object.values(formData).some(
      (value) => value === ''
    )

    if (hasEmptyField) {
      setMessage('Please enter all eight patient details.')
      return
    }


    // --------------------------------------------------------
    // STEP 3: CONVERT INPUT STRINGS INTO NUMBERS
    // --------------------------------------------------------

    const numericData = Object.fromEntries(
      Object.entries(formData).map(([key, value]) => [
        key,
        Number(value),
      ])
    )


    // --------------------------------------------------------
    // STEP 4: VALIDATE NUMERIC VALUES
    // --------------------------------------------------------

    const hasInvalidValue = inputFields.some((field) => {

      const value = numericData[field.name]

      return (
        !Number.isFinite(value) ||
        value < field.min ||
        value > field.max
      )
    })

    if (hasInvalidValue) {
      setMessage('Please check the allowed range for each field.')
      return
    }


    // --------------------------------------------------------
    // STEP 5: PREPARE THE API REQUEST
    // --------------------------------------------------------

    // Convert React field names to the backend's snake_case format.
    const apiPayload = prepareApiPayload(numericData)

    // Activate the loading state while the API processes the data.
    setLoading(true)


    try {

      // ------------------------------------------------------
      // STEP 6: SEND PATIENT DATA TO FASTAPI
      // ------------------------------------------------------

      const response = await fetch(
        `${API_BASE_URL}/predictions/predict`,
        {
          method: 'POST',

          // JSON tells FastAPI that the request body contains JSON.
          headers: {
            'Content-Type': 'application/json',

            // Bearer authentication provides the logged-in user's JWT.
            Authorization: `Bearer ${authToken}`,
          },

          // Send the eight validated patient features.
          body: JSON.stringify(apiPayload),
        }
      )


      // ------------------------------------------------------
      // STEP 7: READ THE BACKEND RESPONSE
      // ------------------------------------------------------
      const data = await response.json()


      // ------------------------------------------------------
      // STEP 8: HANDLE API ERRORS
      // ------------------------------------------------------

      if (!response.ok) {
        if (response.status === 401) {
          onLogout()
          return
        }

        // FastAPI validation errors may return an array of objects.
        // Convert those errors into readable frontend messages.

        let errorMessage = 'Unable to generate prediction.'

        if (Array.isArray(data.detail)) {
          errorMessage = data.detail
            .map((error) => error.msg)
            .join(', ')
        } else if (typeof data.detail === 'string') {
          errorMessage = data.detail
        }

        throw new Error(errorMessage)
      }


      // ------------------------------------------------------
      // STEP 9: VERIFY THE RESPONSE STRUCTURE
      // ------------------------------------------------------

      // Check that the required prediction values are present.
      const hasValidResponse =
        [0, 1].includes(data.predicted_class) &&
        Number.isFinite(data.probability_class_0) &&
        Number.isFinite(data.probability_class_1) &&
        Number.isFinite(data.predicted_class_probability) &&
        data.explanation &&
        data.explanation.feature_contributions

      if (!hasValidResponse) {
        throw new Error('The backend returned an unexpected prediction response.')
      }


      // ------------------------------------------------------
      // STEP 10: SAVE THE REAL PREDICTION RESULT
      // ------------------------------------------------------

      // Store the response from the actual trained ML model.
      // React automatically displays the result when state changes.

      setPredictionResult(data)

      setMessage('Prediction generated successfully.')


    } catch (error) {

      // Display network, authentication, validation or server errors.
      setMessage(
        error.message || 'Something went wrong. Please try again.'
      )

    } finally {

      // Stop the loading indicator whether the request succeeds or fails.
      setLoading(false)
    }
  }


  // ==========================================================
  // 7. RESET THE PREDICTION FORM
  // ==========================================================

  function handleReset() {

    // Restore all input fields to their original empty values.
    setFormData(initialFormData)

    // Remove previous messages and prediction results.
    setMessage('')
    setPredictionResult(null)
  }


  // ==========================================================
  // 8. PREPARE SHAP EXPLANATION DATA
  // ==========================================================

  // Convert the SHAP feature contribution object into an array.
  // Sorting makes the largest absolute contributions appear first.

  const shapFeatures = predictionResult
    ? Object.entries(
        predictionResult.explanation.feature_contributions
      )
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : []


  // Find the largest absolute SHAP contribution.
  // This value is used to calculate relative bar widths.

  const maximumShapValue = Math.max(
    0.000001,
    ...shapFeatures.map(([_, value]) => Math.abs(value))
  )


  // ==========================================================
  // 9. RENDER THE PREDICTION PAGE
  // ==========================================================

  return (

    <section className="prediction-page">

      {/* Page introduction */}
      <div className="prediction-intro">

        <div>

          <h2>Diabetes Risk Prediction</h2>

          <p>
            Enter the required patient health information
            to evaluate diabetes risk using the trained
            machine learning model.
          </p>

        </div>

        <span className="prediction-tag">
          8 Input Features
        </span>

      </div>


      {/* Patient information form */}
      <form
        className="prediction-form"
        onSubmit={handleSubmit}
      >

        <div className="prediction-form-heading">

          <h3>Patient Information</h3>

          <p>
            All eight fields are required for this prediction.
          </p>

        </div>


        {/* Generate all eight input fields */}
        <div className="prediction-input-grid">

          {inputFields.map((field) => (

            <div
              className="prediction-field"
              key={field.name}
            >

              <label htmlFor={field.name}>
                {field.label}
              </label>

              <div className="input-wrapper">

                <input
                  id={field.name}
                  type="number"
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleInputChange}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  placeholder={field.placeholder}
                  required
                  disabled={loading}
                />

                <span className="input-unit">
                  {field.unit}
                </span>

              </div>

            </div>

          ))}

        </div>


        {/* Display validation messages and API feedback */}
        {message && (

          <div
            className="prediction-message"
            role="status"
          >
            {message}
          </div>

        )}


        {/* Form action buttons */}
        <div className="prediction-actions">

          <button
            type="button"
            className="reset-button"
            onClick={handleReset}
            disabled={loading}
          >
            Reset Form
          </button>


          <button
            type="submit"
            className="predict-button"
            disabled={loading}
          >

            {loading
              ? 'Analyzing Patient Data...'
              : 'Predict Diabetes Risk'}

          </button>

        </div>


        <p className="prediction-disclaimer">
          Academic research tool only. This system is not a
          substitute for professional medical diagnosis.
        </p>

      </form>


      {/* ======================================================
          10. ACTUAL MACHINE LEARNING PREDICTION RESULT
          ====================================================== */}

      {predictionResult && (

        <section className="prediction-result-panel">

          <div className="prediction-result-heading">

            <h3>Prediction Result</h3>

            <p>
              Generated by the connected Logistic Regression model.
            </p>

          </div>


          {/* Predicted class result */}
          <div className="prediction-result-card">

            <span className="result-label">
              MODEL PREDICTION
            </span>

            <h2>
              {predictionResult.predicted_class === 1
                ? 'Higher Predicted Risk'
                : 'Lower Predicted Risk'}
            </h2>

            <p>
              Predicted class: {predictionResult.predicted_class}
            </p>

          </div>


          {/* Probability cards */}
          <div className="prediction-probability-grid">

            <div className="prediction-probability-card">

              <span>Class 0 Probability</span>

              <h3>
                {(
                  predictionResult.probability_class_0 * 100
                ).toFixed(2)}%
              </h3>

            </div>


            <div className="prediction-probability-card">

              <span>Class 1 Probability</span>

              <h3>
                {(
                  predictionResult.probability_class_1 * 100
                ).toFixed(2)}%
              </h3>

            </div>


            <div className="prediction-probability-card">

              <span>Predicted Class Probability</span>

              <h3>
                {(
                  predictionResult.predicted_class_probability * 100
                ).toFixed(2)}%
              </h3>

            </div>

          </div>


          {/* ==================================================
              11. INDIVIDUAL SHAP EXPLANATION
              ================================================== */}

          <div className="prediction-shap-panel">

            <div className="prediction-shap-heading">

              <h3>Explainable AI Analysis</h3>

              <p>
                SHAP feature contributions for this prediction.
              </p>

            </div>


            {/* SHAP baseline and model output */}
            <div className="shap-model-values">

              <div>
                <span>Baseline Value</span>

                <strong>
                  {predictionResult.explanation.base_value.toFixed(4)}
                </strong>
              </div>

              <div>
                <span>Model Output</span>

                <strong>
                  {predictionResult.explanation.model_output.toFixed(4)}
                </strong>
              </div>

            </div>

            {/* SHAP graph legend */}
            <div className="shap-chart-legend">

              <div className="shap-legend-item">
                <span className="shap-legend-dot shap-legend-negative" />
                <span>Negative contribution</span>
              </div>

              <div className="shap-legend-item">
                <span className="shap-legend-dot shap-legend-zero" />
                <span>Zero contribution</span>
              </div>

              <div className="shap-legend-item">
                <span className="shap-legend-dot shap-legend-positive" />
                <span>Positive contribution</span>
              </div>

            </div>

            {/* SHAP graph direction labels */}
            <div className="shap-axis-labels">
              <span>← Decreases model output</span>
              <span>0</span>
              <span>Increases model output →</span>
            </div>




            {/* Display the eight feature contributions */}
            <div className="shap-feature-list">

              {shapFeatures.map(([featureName, contribution]) => {

                // Calculate the contribution bar width relative
                // to the largest absolute SHAP contribution.
                const barWidth =
                  (Math.abs(contribution) / maximumShapValue) * 100

                // Positive values increase the model output.
                // Negative values decrease the model output.
                const isPositive = contribution >= 0

                return (

                  <div
                    className="shap-feature-item"
                    key={featureName}
                  >

                    <div className="shap-feature-info">

                      <span>
                        {featureName}
                      </span>

                      <strong>
                        {contribution >= 0 ? '+' : ''}
                        {contribution.toFixed(4)}
                      </strong>

                    </div>



                    {/* SHAP contribution graph */}
                    <div className="shap-bar-background">

                      {/* Central zero reference line */}
                      <div className="shap-zero-line" />

                      {/* Contribution bar: negative left, positive right */}
                      <div
                        className={
                          isPositive
                            ? 'shap-bar shap-positive'
                            : 'shap-bar shap-negative'
                        }
                        style={{
                          width: `${barWidth / 2}%`,
                          left: isPositive
                            ? '50%'
                            : `calc(50% - ${barWidth / 2}%)`,
                        }}
                      />

                    </div>



                    <small>
                      {isPositive
                        ? 'Positive contribution to model output'
                        : 'Negative contribution to model output'}
                    </small>

                  </div>

                )
              })}

            </div>

            <p className="shap-disclaimer">
              SHAP values explain how features contribute to the
              model output (log-odds). They do not establish
              medical causation.
            </p>

          </div>

        </section>

      )}

    </section>
  )
}


// Export the component so App.jsx can use it.
export default Prediction