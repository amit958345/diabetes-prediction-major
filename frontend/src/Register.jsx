
/*
============================================================
DIAPREDICT AI - USER REGISTRATION COMPONENT
Major Project: Explainable AI-Based Diabetes Risk Prediction

DESIGN:
- Dark teal AI branding panel on the left.
- Pale blue-grey registration section on the right.
- Matching login page styling and animations.

FUNCTIONALITY:
- Collect name, email and password.
- Confirm password before submission.
- Connect to the FastAPI registration endpoint.
- Display success and error messages.
- Navigate back to Login.

BACKEND:
POST /auth/register

SECURITY:
- Never store passwords in browser storage.
- Backend validates and securely processes registration.
============================================================
*/

import { useState } from 'react'
import { API_BASE_URL } from './config'

// ==========================================================
// REGISTRATION COMPONENT
// ==========================================================

function Register({ onGoToLogin }) {

  // Store the full name entered by the user.
  const [name, setName] = useState('')

  // Store the email entered by the user.
  const [email, setEmail] = useState('')

  // Store the password entered by the user.
  const [password, setPassword] = useState('')

  // Store the password confirmation.
  const [confirmPassword, setConfirmPassword] = useState('')

  // Track whether the registration request is running.
  const [loading, setLoading] = useState(false)

  // Store validation or backend error messages.
  const [error, setError] = useState('')

  // Store a successful registration message.
  const [success, setSuccess] = useState('')


  // ========================================================
  // HANDLE REGISTRATION FORM SUBMISSION
  // ========================================================

  async function handleRegister(event) {

    // Prevent the browser from refreshing the page.
    event.preventDefault()

    // Clear messages from previous attempts.
    setError('')
    setSuccess('')

    // Remove unnecessary spaces from the name and email.
    const cleanedName = name.trim()
    const cleanedEmail = email.trim()

    // Check that the name contains at least two characters.
    if (cleanedName.length < 2 || cleanedName.length > 100) {
      setError('Name must contain between 2 and 100 characters.')
      return
    }

    // Check whether both password fields match.
    if (password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    // Validate the password length before contacting FastAPI.
    if (password.length < 8 || password.length > 128) {
      setError('Password must contain between 8 and 128 characters.')
      return
    }

    // Start the loading state.
    setLoading(true)

    try {

      // Send registration information to the FastAPI backend.
      const response = await fetch(
        `${API_BASE_URL}/auth/register`,
        {
          method: 'POST',

          // Tell FastAPI that the request body contains JSON.
          headers: {
            'Content-Type': 'application/json',
          },

          // Send the fields expected by the UserCreate schema.
          body: JSON.stringify({
            name: cleanedName,
            email: cleanedEmail,
            password: password,
          }),
        }
      )

      // Convert the backend response into a JavaScript object.
      const data = await response.json()

      // Handle registration errors returned by FastAPI.
      if (!response.ok) {
        const detail = data.detail

        // Handle both string and validation-list error responses.
        const message = Array.isArray(detail)
          ? detail.map((item) => item.msg).join(', ')
          : detail

        throw new Error(
          message || 'Registration failed. Please try again.'
        )
      }

      // Display a success message.
      setSuccess('Account created successfully. You can now sign in.')

      // Clear all registration fields after successful creation.
      setName('')
      setEmail('')
      setPassword('')
      setConfirmPassword('')

    } catch (error) {

      // Show a useful connection or registration error.
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
  // REGISTRATION PAGE INTERFACE
  // ========================================================

  return (

    <main className="auth-page auth-pale-theme auth-register-theme">

      {/* Main split-screen registration container. */}
      <section className="auth-split-container">


        {/* ==================================================
            LEFT SIDE - AI BRANDING PANEL
            ================================================== */}

        <section className="auth-left-panel">

          {/* Application logo and brand name. */}
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


          {/* Main project introduction. */}
          <div className="auth-left-content">

            <span className="auth-left-eyebrow">
              EXPLAINABLE ARTIFICIAL INTELLIGENCE
            </span>

            <h2>
              Start Your
              <br />
              <span>Health Journey.</span>
            </h2>

            <p>
              Create your account and explore intelligent
              diabetes risk prediction powered by machine
              learning and explainable AI insights.
            </p>

          </div>


          {/* Decorative AI visualization.
              This is a design element, not a real prediction. */}
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


          {/* Branding panel footer. */}
          <div className="auth-left-footer">

            <span className="auth-status-dot"></span>

            AI-POWERED RESEARCH PLATFORM

          </div>

        </section>


        {/* ==================================================
            RIGHT SIDE - REGISTRATION FORM
            ================================================== */}

        <section className="auth-right-panel">

          {/* Registration card. */}
          <div className="auth-login-card auth-register-card">


            {/* Registration heading. */}
            <div className="auth-heading">

              <span className="auth-form-eyebrow">
                CREATE YOUR ACCOUNT
              </span>

              <h2>Create Account</h2>

              <p>
                Join DiaPredict AI to access your health dashboard.
              </p>

            </div>


            {/* Registration form. */}
            <form onSubmit={handleRegister}>


              {/* Full name field. */}
              <div className="auth-field">

                <label htmlFor="register-name">
                  Full Name
                </label>

                <input
                  id="register-name"
                  type="text"
                  placeholder="Enter your full name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  minLength={2}
                  maxLength={100}
                  autoComplete="name"
                  required
                />

              </div>


              {/* Email field. */}
              <div className="auth-field">

                <label htmlFor="register-email">
                  Email Address
                </label>

                <input
                  id="register-email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  autoComplete="email"
                  required
                />

              </div>


              {/* Password field. */}
              <div className="auth-field">

                <label htmlFor="register-password">
                  Password
                </label>

                <input
                  id="register-password"
                  type="password"
                  placeholder="Create a password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  minLength={8}
                  maxLength={128}
                  autoComplete="new-password"
                  required
                />

              </div>


              {/* Confirm password field. */}
              <div className="auth-field">

                <label htmlFor="confirm-password">
                  Confirm Password
                </label>

                <input
                  id="confirm-password"
                  type="password"
                  placeholder="Re-enter your password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  minLength={8}
                  maxLength={128}
                  autoComplete="new-password"
                  required
                />

              </div>


              {/* Display backend or validation errors. */}
              {error && (
                <div className="auth-error" role="alert">
                  {error}
                </div>
              )}


              {/* Display successful registration. */}
              {success && (
                <div className="auth-success" role="status">
                  {success}
                </div>
              )}


              {/* Submit registration form. */}
              <button
                type="submit"
                className="auth-submit-button"
                disabled={loading}
              >
                {loading ? 'Creating Account...' : 'Create Account'}

                {!loading && (
                  <span className="auth-button-arrow">→</span>
                )}
              </button>

            </form>


            {/* Navigate back to Login. */}
            <div className="auth-switch">

              <p>
                Already have an account?

                <button
                  type="button"
                  onClick={onGoToLogin}
                >
                  Sign In
                </button>
              </p>

            </div>


            {/* Registration security information. */}
            <div className="auth-security-note">
              <span>◈</span>
              <p>Your account is protected by secure authentication.</p>
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


// Export Register so App.jsx can display it.
export default Register
