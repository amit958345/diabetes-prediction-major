/*
============================================================
DIAPREDICT AI - LOGIN COMPONENT
Major Project: Explainable AI-Based Diabetes Risk Prediction

DESIGN:
- Full-height split-screen authentication.
- Dark teal AI branding panel on the left.
- Pale blue-grey login section on the right.
- Compact login card with soft shadows.

FUNCTIONALITY:
- FastAPI login integration.
- JWT response handling.
- Email and password validation.
- Loading and error states.
============================================================
*/

import { useState } from 'react'
import { API_BASE_URL } from './config'

// ==========================================================
// LOGIN COMPONENT
// ==========================================================

function Login({ onLoginSuccess, onGoToRegister }) {

  // Store the email entered by the user.
  const [email, setEmail] = useState('')

  // Store the password entered by the user.
  const [password, setPassword] = useState('')

  // Track whether the backend request is running.
  const [loading, setLoading] = useState(false)

  // Store backend or connection error messages.
  const [error, setError] = useState('')


  // ========================================================
  // HANDLE LOGIN FORM SUBMISSION
  // ========================================================

  async function handleLogin(event) {

    // Prevent the browser from refreshing the page.
    event.preventDefault()

    // Clear any previous error.
    setError('')

    // Activate the loading state.
    setLoading(true)

    try {

      // Send the credentials to the FastAPI login endpoint.
      const response = await fetch(
        `${API_BASE_URL}/auth/login`,
        {
          method: 'POST',

          // Tell the backend that the request body is JSON.
          headers: {
            'Content-Type': 'application/json',
          },

          // Convert the form values into JSON.
          body: JSON.stringify({
            email: email.trim(),
            password: password,
          }),
        }
      )

      // Read the JSON response from FastAPI.
      const data = await response.json()

      // Handle incorrect credentials or backend errors.
      if (!response.ok) {
        throw new Error(
          data.detail || 'Login failed. Please try again.'
        )
      }

      // Send the JWT token and user information to App.jsx.
      onLoginSuccess(data)

    } catch (error) {

      // Display a helpful connection or authentication error.
      setError(
        error.message === 'Failed to fetch'
          ? 'Cannot connect to the server. Check whether FastAPI is running.'
          : error.message
      )

    } finally {

      // Stop loading after success or failure.
      setLoading(false)
    }
  }


  // ========================================================
  // LOGIN PAGE INTERFACE
  // ========================================================

  return (

    <main className="auth-page auth-pale-theme">

      {/* Main split-screen authentication container. */}
      <section className="auth-split-container">


        {/* ==================================================
            LEFT SIDE - AI BRANDING PANEL
            ================================================== */}

        <section className="auth-left-panel">

          {/* Application branding. */}
          <div className="auth-left-brand">

            <div className="auth-logo">
              D
            </div>

            <div>
              <h1 className="auth-brand-title">
                <span className="brand-name">DiaPredict</span>
                <span className="brand-ai"> AI</span>
              </h1>

              <p>Intelligent Health Analytics</p>
            </div>

          </div>


          {/* Main branding message. */}
          <div className="auth-left-content">

            <span className="auth-left-eyebrow">
              EXPLAINABLE ARTIFICIAL INTELLIGENCE
            </span>

            <h2>
              Your AI-Powered
              <br />
              <span>Health Assistant</span>
            </h2>

            <p>
              Explore diabetes risk prediction through
              intelligent machine learning and explainable
              AI-based insights.
            </p>

          </div>


          {/* Decorative AI visualization.
              This is visual decoration, not a real prediction. */}
          <div className="auth-health-visual" aria-hidden="true">

            <div className="health-visual-ring ring-one"></div>

            <div className="health-visual-ring ring-two"></div>

            <div className="health-visual-core">
              <span>AI</span>
              <small>HEALTH ANALYTICS</small>
            </div>

            <span className="health-visual-label label-ml">
              ML
            </span>

            <span className="health-visual-label label-shap">
              SHAP
            </span>

            <span className="health-visual-label label-data">
              DATA
            </span>

          </div>


          {/* Left panel footer. */}
          <div className="auth-left-footer">

            <span className="auth-status-dot"></span>

            AI-POWERED RESEARCH PLATFORM

          </div>

        </section>


        {/* ==================================================
            RIGHT SIDE - LOGIN SECTION
            ================================================== */}

        <section className="auth-right-panel">

          {/* Compact login card. */}
          <div className="auth-login-card">


            {/* Login heading. */}
            <div className="auth-heading">

              <span className="auth-form-eyebrow">
                SECURE ACCOUNT ACCESS
              </span>

              <h2>Welcome Back</h2>

              <p>
                Sign in to access your health dashboard.
              </p>

            </div>


            {/* Login form. */}
            <form onSubmit={handleLogin}>


              {/* Email input field. */}
              <div className="auth-field">

                <label htmlFor="login-email">
                  Email Address
                </label>

                <input
                  id="login-email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  required
                />

              </div>


              {/* Password input field. */}
              <div className="auth-field">

                <label htmlFor="login-password">
                  Password
                </label>

                <input
                  id="login-password"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete="current-password"
                  minLength={8}
                  maxLength={128}
                  required
                />

              </div>


              {/* Display errors only when an error exists. */}
              {error && (
                <div className="auth-error" role="alert">
                  {error}
                </div>
              )}


              {/* Login submission button. */}
              <button
                type="submit"
                className="auth-submit-button"
                disabled={loading}
              >
                {loading ? 'Signing In...' : 'Sign In'}

                {!loading && (
                  <span className="auth-button-arrow">→</span>
                )}
              </button>

            </form>


            {/* Registration navigation. */}
            <div className="auth-switch">

              <p>
                Don't have an account?

                <button
                  type="button"
                  onClick={onGoToRegister}
                >
                  Create Account
                </button>
              </p>

            </div>


            {/* Security information. */}
            <div className="auth-security-note">
              <span>◈</span>
              <p>Secure authentication powered by JWT.</p>
            </div>


            {/* Project identification. */}
            <div className="auth-footer">
              Explainable AI-Based Diabetes Risk Prediction
            </div>

          </div>

        </section>

      </section>

    </main>
  )
}


// Export the component for use inside App.jsx.
export default Login