/*
============================================================
DIAPREDICT - PREMIUM AI HEALTHCARE APPLICATION
Project: Explainable AI-Based Diabetes Risk Prediction
Design: Next-Generation Hybrid AI

PURPOSE:
1. Manage Login and Registration.
2. Manage authenticated user sessions.
3. Display the research dashboard.
4. Navigate between application pages.
5. Pass the JWT token to the Prediction component.
6. Fetch real prediction history from FastAPI.
7. Display real dashboard statistics.
8. Search and filter prediction records.

BACKEND APIs USED:
POST /auth/login
POST /auth/register
GET  /predictions/history
GET  /predictions/dashboard

IMPORTANT:
Dashboard research metrics are based on the Phase 10
independent test evaluation.

The application is an academic research tool and is not
a substitute for professional medical diagnosis.
============================================================
*/


// ==========================================================
// 1. REACT IMPORTS
// ==========================================================

// useState stores application data and UI state.
// useEffect runs API requests when selected state changes.
import { useState, useEffect } from 'react'
import { API_BASE_URL } from './config';

// Existing patient prediction component.
import Prediction from './Prediction.jsx'

// User login component.
import Login from './Login.jsx'

// User registration component.
import Register from './Register.jsx'

// Import the global SHAP feature-importance visualization component.
// This component fetches real feature-importance values from FastAPI.
import GlobalShapImportance from './GlobalShapImportance.jsx';

// Import the individual SHAP waterfall visualization.
// This component displays explanations for selected predictions.
import ShapWaterfall from './ShapWaterfall.jsx';

// Import the SHAP beeswarm visualization component.
// This component displays individual SHAP contributions
// using actual data retrieved from the FastAPI backend.
import ShapBeeswarm from './ShapBeeswarm.jsx';

// Import the SHAP dependence visualization component.
import ShapDependence from './ShapDependence.jsx';
import ModelInsights from './ModelInsights.jsx';

// Main application styling.
import './App.css'


// ==========================================================
// 2. SIDEBAR NAVIGATION CONFIGURATION
// ==========================================================

// Defines all navigation items displayed in the sidebar.
// id identifies the page.
// icon represents the navigation symbol.
// label is the displayed page name.
// group determines the navigation section.

const navigationItems = [
  {
    id: 'dashboard',
    icon: '▦',
    label: 'Dashboard',
    group: 'WORKSPACE',
  },
  {
    id: 'prediction',
    icon: '✚',
    label: 'New Prediction',
    group: 'WORKSPACE',
  },
  {
    id: 'history',
    icon: '◷',
    label: 'Prediction History',
    group: 'WORKSPACE',
  },
  {
    id: 'analytics',
    icon: '▥',
    label: 'Advanced Analytics',
    group: 'WORKSPACE',
  },
  {
    id: 'insights',
    icon: '◫',
    label: 'Model Insights',
    group: 'INTELLIGENCE',
  },
  {
    id: 'explainability',
    icon: '◈',
    label: 'Explainable AI',
    group: 'INTELLIGENCE',
  },
  {
    id: 'reports',
    icon: '▤',
    label: 'Research Reports',
    group: 'INTELLIGENCE',
  },
  {
    id: 'about',
    icon: 'ⓘ',
    label: 'About Project',
    group: 'SYSTEM',
  },
  {
    id: 'admin',
    icon: '♙',
    label: 'User Management',
    group: 'ADMINISTRATION',
    adminOnly: true,
  },
]


// ==========================================================
// 3. RESEARCH METRICS
// These values come from the Phase 10 independent test.
// ==========================================================

// These are research metrics, not live database values.
// Total Predictions will be loaded from the backend.

const dashboardStatistics = [
  {
    title: 'Total Predictions',
    value: '—',
    description: 'Live database statistics',
    icon: '◷',
    theme: 'cyan',
    tag: 'LIVE DATA',
  },
  {
    title: 'Test Accuracy',
    value: '72.08%',
    description: 'Independent test evaluation',
    icon: '◎',
    theme: 'blue',
    tag: 'RESEARCH',
  },
  {
    title: 'Dataset Records',
    value: '768',
    description: 'Pima Indians Diabetes Dataset',
    icon: '▤',
    theme: 'purple',
    tag: 'DATASET',
  },
  {
    title: 'Active Algorithm',
    value: 'Logistic Regression',
    description: 'Selected research model',
    icon: '◉',
    theme: 'green',
    tag: 'ML MODEL',
  },
]


// ==========================================================
// 4. REUSABLE STATISTIC CARD
// ==========================================================

// Displays one dashboard statistic.
// Props are received from the dashboardStatistics array.

function StatisticCard({
  title,
  value,
  description,
  icon,
  theme,
  tag,
}) {
  return (
    <article className={`premium-stat-card ${theme}`}>

      {/* Decorative background glow */}
      <div className="stat-glow"></div>

      {/* Card top section */}
      <div className="premium-stat-top">

        <div className="premium-stat-icon">
          {icon}
        </div>

        <span className="premium-stat-tag">
          {tag}
        </span>

      </div>

      {/* Statistic information */}
      <div className="premium-stat-body">

        <p className="premium-stat-title">
          {title}
        </p>

        <h2 className="premium-stat-value">
          {value}
        </h2>

        <p className="premium-stat-description">
          {description}
        </p>

      </div>

    </article>
  )
}


// ==========================================================
// 5. REUSABLE SECTION HEADING
// ==========================================================

// Creates a consistent heading for dashboard panels.

function SectionHeading({
  eyebrow,
  title,
  description,
}) {
  return (
    <div className="premium-section-heading">

      <div>

        <p className="section-eyebrow">
          {eyebrow}
        </p>

        <h3>
          {title}
        </h3>

        <p className="section-description">
          {description}
        </p>

      </div>

    </div>
  )
}


// ==========================================================
// 6. MAIN APPLICATION COMPONENT
// ==========================================================

function App() {

  // ========================================================
  // 6.1 AUTHENTICATION STATE
  // ========================================================

  // Controls whether Login or Register is displayed.
  const [authView, setAuthView] = useState('login')


  // Restore the previously saved JWT token.
  // The token is required for authenticated API requests.
  const [authToken, setAuthToken] = useState(() => {
    return localStorage.getItem('diapredict_access_token')
  })


  // Restore the previously saved user profile.
  // Passwords must never be stored in localStorage.
  const [currentUser, setCurrentUser] = useState(() => {

    try {
      const savedUser = localStorage.getItem('diapredict_user')

      return savedUser ? JSON.parse(savedUser) : null

    } catch {
      return null
    }

  })


  // ========================================================
  // 6.2 APPLICATION NAVIGATION STATE
  // ========================================================

  // Determines which application page is currently visible.
  const [activePage, setActivePage] = useState('dashboard')


  // ========================================================
  // 6.2.1 EXPLAINABLE AI TAB STATE
  // ========================================================

  // Stores the currently selected SHAP visualization tab.
  // The Global Importance graph is selected by default.
  const [activeShapTab, setActiveShapTab] = useState('global')

  // Independent tab state for Research Reports and About Project.
  const [activeReportTab, setActiveReportTab] = useState('overview')
  const [activeAboutTab, setActiveAboutTab] = useState('overview')


  // ========================================================
  // 6.3 PREDICTION HISTORY STATE
  // ========================================================

  // Stores actual prediction records returned by FastAPI.
  const [predictionHistory, setPredictionHistory] = useState([])

  // Stores actual dashboard statistics returned by FastAPI.
  const [historyDashboard, setHistoryDashboard] = useState(null)

  // Advanced analytics returned by the authenticated analytics API.
  const [analyticsData, setAnalyticsData] = useState(null)
  const [analyticsLoading, setAnalyticsLoading] = useState(false)
  const [analyticsError, setAnalyticsError] = useState('')

  // Tracks whether history information is loading.
  const [historyLoading, setHistoryLoading] = useState(false)

  // Stores API errors.
  const [historyError, setHistoryError] = useState('')

  // Administrator-only user management data.
  const [adminUsers, setAdminUsers] = useState([])
  const [adminDashboard, setAdminDashboard] = useState(null)
  const [adminLoading, setAdminLoading] = useState(false)
  const [adminError, setAdminError] = useState('')
  const [adminUserSearch, setAdminUserSearch] = useState('')

  // Stores the search text entered by the user.
  const [historySearch, setHistorySearch] = useState('')

  // Stores the selected prediction filter.
  // Allowed values: all, positive, negative.
  const [historyFilter, setHistoryFilter] = useState('all')

  // Stores the prediction whose complete details are currently open.
  const [selectedHistoryPredictionId, setSelectedHistoryPredictionId] = useState(null)

  // Tracks the prediction report currently being downloaded.
  const [downloadingReportId, setDownloadingReportId] = useState(null)

  // Displays errors from the PDF download request.
  const [reportDownloadError, setReportDownloadError] = useState('')

  // Tracks the complete Research Report PDF download.
  const [downloadingResearchReport, setDownloadingResearchReport] = useState(false)
  const [researchReportDownloadError, setResearchReportDownloadError] = useState('')

  // Tracks whether the CSV export request is in progress.
  const [downloadingCsv, setDownloadingCsv] = useState(false)

  // Displays errors from the CSV download request.
  const [csvDownloadError, setCsvDownloadError] = useState('')


  // ========================================================
  // 6.4 FETCH AUTHENTICATED PREDICTION HISTORY
  // ========================================================

  // This effect runs when:
  // 1. The user opens the History page.
  // 2. The authentication token changes.
  //
  // It requests both:
  // GET /predictions/history
  // GET /predictions/dashboard

  useEffect(() => {

    // Do not fetch history when another page is active.
    if (activePage !== 'history') {
      return
    }

    // A JWT token is required for both API endpoints.
    if (!authToken) {
      setHistoryError('Authentication required. Please log in again.')
      return
    }

    // Allows pending API requests to be cancelled.
    const controller = new AbortController()


    // Asynchronous function for fetching backend data.
    const fetchHistoryData = async () => {

      setHistoryLoading(true)
      setHistoryError('')

      try {

        // Attach the JWT token to the request.
        const requestOptions = {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${authToken}`,
            'Content-Type': 'application/json',
          },
          signal: controller.signal,
        }


        // Request the authenticated user's prediction history.
        const historyResponse = await authenticatedFetch(
          `${API_BASE_URL}/predictions/history`,
          requestOptions
        )


        // Request the user's dashboard statistics.
        const dashboardResponse = await authenticatedFetch(
          `${API_BASE_URL}/predictions/dashboard`,
          requestOptions
        )


        // Convert backend JSON responses into JavaScript objects.
        const historyData = await historyResponse.json()
        const dashboardData = await dashboardResponse.json()


        // Check whether either request failed.
        if (!historyResponse.ok || !dashboardResponse.ok) {

          const errorMessage =
            historyData.detail ||
            dashboardData.detail ||
            'Unable to retrieve prediction history.'

          throw new Error(errorMessage)
        }


        // Save actual records returned by FastAPI.
        setPredictionHistory(
          Array.isArray(historyData) ? historyData : []
        )


        // Save actual dashboard statistics.
        setHistoryDashboard(dashboardData)

      } catch (error) {

        // Ignore requests intentionally cancelled by React.
        if (error.name === 'AbortError') {
          return
        }

        // Display the actual backend error.
        setHistoryError(error.message)

      } finally {

        // Stop loading if the request was not cancelled.
        if (!controller.signal.aborted) {
          setHistoryLoading(false)
        }

      }

    }


    // Start fetching history information.
    fetchHistoryData()


    // Cancel pending requests when leaving the page.
    return () => {
      controller.abort()
    }

  }, [activePage, authToken])

    // ========================================================
    // 6.4.1 FETCH AUTHENTICATED DASHBOARD STATISTICS
    // ========================================================

    // This effect runs when the Dashboard page opens
    // or when the authentication token changes.
    useEffect(() => {

        // Do not fetch dashboard data on other pages.
        if (activePage !== 'dashboard') {
            return
        }

        // Check whether the user is authenticated.
        if (!authToken) {
            setHistoryDashboard(null)
            return
        }

        // Create a controller to cancel pending API requests.
        const controller = new AbortController()

        // Fetch dashboard statistics from the FastAPI backend.
        const fetchDashboardData = async () => {

            try {

                // Send an authenticated GET request.
                const response = await authenticatedFetch(
                    `${API_BASE_URL}/predictions/dashboard`,
                    {
                        method: 'GET',

                        // Include the JWT authentication token.
                        headers: {
                            'Accept': 'application/json',
                            'Authorization': `Bearer ${authToken}`,
                        },

                        // Allow the request to be cancelled.
                        signal: controller.signal,
                    }
                )

                // Check whether the API request succeeded.
                if (!response.ok) {
                    throw new Error(
                        `Dashboard API error: ${response.status}`
                    )
                }

                // Convert the JSON response into JavaScript data.
                const dashboardData = await response.json()

                // Store the actual dashboard statistics.
                setHistoryDashboard(dashboardData)

            } catch (error) {

                // Ignore intentionally cancelled requests.
                if (error.name === 'AbortError') {
                    return
                }

                // Display API errors in the browser console.
                console.error(
                    'Unable to retrieve dashboard statistics:',
                    error
                )

            }
        }

        // Start fetching dashboard statistics.
        fetchDashboardData()

        // Cancel the request when leaving the Dashboard page.
        return () => {
            controller.abort()
        }

    }, [activePage, authToken])
  // ========================================================
  // ========================================================
  // 6.4.2 FETCH ADVANCED ANALYTICS
  // ========================================================

  useEffect(() => {
    if (activePage !== 'analytics') return

    if (!authToken) {
      setAnalyticsError('Authentication required. Please log in again.')
      setAnalyticsData(null)
      return
    }

    const controller = new AbortController()

    const fetchAnalytics = async () => {
      setAnalyticsLoading(true)
      setAnalyticsError('')

      try {
        const response = await authenticatedFetch(
          `${API_BASE_URL}/predictions/analytics`,
          {
            method: 'GET',
            headers: {
              Accept: 'application/json',
              Authorization: `Bearer ${authToken}`,
            },
            signal: controller.signal,
          }
        )

        const data = await response.json()

        if (!response.ok) {
          throw new Error(data?.detail || `Analytics API error: ${response.status}`)
        }

        setAnalyticsData(data)
      } catch (error) {
        if (error.name === 'AbortError') return
        setAnalyticsError(error.message || 'Unable to load advanced analytics.')
      } finally {
        if (!controller.signal.aborted) {
          setAnalyticsLoading(false)
        }
      }
    }

    fetchAnalytics()

    return () => controller.abort()
  }, [activePage, authToken])

  // ========================================================
  // 6.4.3 FETCH ADMIN DASHBOARD AND USER LIST
  // ========================================================
  // These endpoints are protected by the backend require_admin dependency.
  // The API returns only safe user fields: id, name, email and role.
  useEffect(() => {
    if (activePage !== 'admin') return

    if (currentUser?.role !== 'admin') {
      setAdminUsers([])
      setAdminDashboard(null)
      setAdminError('Administrator access is required to view User Management.')
      setActivePage('dashboard')
      return
    }

    if (!authToken) {
      setAdminError('Authentication required. Please log in again.')
      return
    }

    const controller = new AbortController()

    const fetchAdminData = async () => {
      setAdminLoading(true)
      setAdminError('')

      try {
        const [dashboardResponse, usersResponse] = await Promise.all([
          authenticatedFetch(`${API_BASE_URL}/admin/dashboard`, {
            method: 'GET',
            headers: { Accept: 'application/json' },
            signal: controller.signal,
          }),
          authenticatedFetch(`${API_BASE_URL}/admin/users?skip=0&limit=100`, {
            method: 'GET',
            headers: { Accept: 'application/json' },
            signal: controller.signal,
          }),
        ])

        const dashboardData = await dashboardResponse.json()
        const usersData = await usersResponse.json()

        if (!dashboardResponse.ok) {
          throw new Error(dashboardData?.detail || `Admin dashboard error: ${dashboardResponse.status}`)
        }

        if (!usersResponse.ok) {
          throw new Error(usersData?.detail || `User list error: ${usersResponse.status}`)
        }

        setAdminDashboard(dashboardData)
        setAdminUsers(Array.isArray(usersData) ? usersData : [])
      } catch (error) {
        if (error.name === 'AbortError') return
        setAdminError(error.message || 'Unable to load administrator data.')
      } finally {
        if (!controller.signal.aborted) {
          setAdminLoading(false)
        }
      }
    }

    fetchAdminData()
    return () => controller.abort()
  }, [activePage, authToken, currentUser?.role])


  // 6.5 FILTER PREDICTION HISTORY
  // ========================================================

  // This variable contains only the records matching
  // the user's search text and selected filter.

  const filteredHistory = predictionHistory.filter((record) => {

    // Convert the search text into lowercase.
    const searchValue = historySearch.toLowerCase().trim()


    // Prepare searchable information from each record.
    const searchableText = [
      record.id,
      record.age,
      record.glucose,
      record.bmi,
      record.predicted_class,
      record.created_at,
    ].join(' ').toLowerCase()


    // Check whether the record matches the search text.
    const matchesSearch = searchableText.includes(searchValue)


    // Class 1 represents the positive outcome label.
    // Class 0 represents the negative outcome label.
    const matchesFilter =
      historyFilter === 'all' ||
      (historyFilter === 'positive' && record.predicted_class === 1) ||
      (historyFilter === 'negative' && record.predicted_class === 0)


    return matchesSearch && matchesFilter

  })


  // Find the record currently opened in the Prediction Details view.
  const selectedHistoryRecord = predictionHistory.find(
    (record) => String(record.id) === String(selectedHistoryPredictionId)
  ) || null

  // The backend stores the probability of the predicted class.
  // Convert it into explicit Class 1 and Class 0 probabilities.
  const selectedClass1Probability = selectedHistoryRecord
    ? (Number(selectedHistoryRecord.predicted_class) === 1
        ? Number(selectedHistoryRecord.predicted_probability)
        : 1 - Number(selectedHistoryRecord.predicted_probability))
    : null

  const selectedClass0Probability = selectedHistoryRecord
    ? 1 - selectedClass1Probability
    : null

  const filteredAdminUsers = adminUsers.filter((user) => {
    const query = adminUserSearch.trim().toLowerCase()
    if (!query) return true
    return [user.id, user.name, user.email, user.role]
      .join(' ')
      .toLowerCase()
      .includes(query)
  })


  // ========================================================
  // 6.5.1 DOWNLOAD INDIVIDUAL PREDICTION PDF
  // ========================================================

  // Downloads a report for the selected prediction using the
  // authenticated user's JWT token. The backend verifies ownership.
  const handleDownloadPredictionReport = async (predictionId) => {

    if (!authToken || !predictionId) {
      setReportDownloadError('Authentication or prediction ID is missing.')
      return
    }

    setDownloadingReportId(predictionId)
    setReportDownloadError('')

    try {
      const response = await authenticatedFetch(
        `${API_BASE_URL}/predictions/${predictionId}/report`,
        {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${authToken}`,
            Accept: 'application/pdf',
          },
        }
      )

      if (!response.ok) {
        let message = `Unable to download report (HTTP ${response.status}).`

        try {
          const errorData = await response.json()
          if (errorData?.detail) {
            message = errorData.detail
          }
        } catch {
          // Keep the HTTP status message when the response is not JSON.
        }

        throw new Error(message)
      }

      const pdfBlob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(pdfBlob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = `DiaPredict_Prediction_${predictionId}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(downloadUrl)

    } catch (error) {
      setReportDownloadError(
        error.message || 'An unexpected error occurred while downloading the PDF.'
      )
    } finally {
      setDownloadingReportId(null)
    }
  }


  // ========================================================
  // 6.5.2 DOWNLOAD COMPLETE RESEARCH REPORT AS PDF
  // ========================================================

  // Requests the complete five-section research report from FastAPI.
  const handleDownloadResearchReport = async () => {

    if (!authToken) {
      setResearchReportDownloadError('Authentication token is missing. Please log in again.')
      return
    }

    setDownloadingResearchReport(true)
    setResearchReportDownloadError('')

    try {
      const response = await authenticatedFetch(

        `${API_BASE_URL}/predictions/research-report`,
        {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${authToken}`,
            Accept: 'application/pdf',
          },
        }
      )

      if (!response.ok) {
        let message = `Unable to download Research Report (HTTP ${response.status}).`
        try {
          const errorData = await response.json()
          if (errorData?.detail) message = errorData.detail
        } catch {
          // Keep the HTTP status message if the response is not JSON.
        }
        throw new Error(message)
      }

      const pdfBlob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(pdfBlob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = 'DiaPredict_Research_Report.pdf'
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(downloadUrl)

    } catch (error) {
      setResearchReportDownloadError(
        error.message || 'An unexpected error occurred while downloading the Research Report.'
      )
    } finally {
      setDownloadingResearchReport(false)
    }
  }


  // ========================================================
  // 6.5.2 DOWNLOAD FILTERED PREDICTION HISTORY AS CSV
  // ========================================================

  // Exports the same records currently visible in Prediction History.
  // The backend still enforces ownership using the authenticated JWT.
  const handleDownloadPredictionHistoryCsv = async () => {

    if (!authToken) {
      setCsvDownloadError('Authentication token is missing. Please log in again.')
      return
    }

    setDownloadingCsv(true)
    setCsvDownloadError('')

    try {
      const queryParams = new URLSearchParams()
      const trimmedSearch = historySearch.trim()

      if (trimmedSearch) {
        queryParams.set('search', trimmedSearch)
      }

      if (historyFilter === 'positive') {
        queryParams.set('predicted_class', '1')
      } else if (historyFilter === 'negative') {
        queryParams.set('predicted_class', '0')
      }

      const queryString = queryParams.toString()
      const exportUrl = `${API_BASE_URL}/predictions/history/export${
        queryString ? `?${queryString}` : ''
      }`

      const response = await authenticatedFetch(exportUrl, {
        method: 'GET',
        headers: {
          Authorization: `Bearer ${authToken}`,
          Accept: 'text/csv',
        },
      })

      if (!response.ok) {
        let message = `Unable to export prediction history (HTTP ${response.status}).`

        try {
          const errorData = await response.json()
          if (errorData?.detail) {
            message = errorData.detail
          }
        } catch {
          // Keep the HTTP status message if the response is not JSON.
        }

        throw new Error(message)
      }

      const csvBlob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(csvBlob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = 'DiaPredict_Prediction_History.csv'
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(downloadUrl)

    } catch (error) {
      setCsvDownloadError(
        error.message || 'An unexpected error occurred while exporting CSV.'
      )
    } finally {
      setDownloadingCsv(false)
    }
  }


  // ========================================================
  // 6.6 CURRENT NAVIGATION INFORMATION
  // ========================================================

  // Find the currently selected navigation item.
  const activeItem = navigationItems.find(
    (item) => item.id === activePage
  )

  // Determine the current page title.
  const pageTitle = activeItem
    ? activeItem.label
    : 'Dashboard'


  // ========================================================
  // 6.7 LOGIN SUCCESS HANDLER
  // ========================================================

  // Called after Login.jsx receives a successful response.

  const handleLoginSuccess = (loginData) => {

    // Extract JWT token and user information.
    const token = loginData.access_token
    const user = loginData.user


    // Save the JWT token in localStorage.
    localStorage.setItem(
      'diapredict_access_token',
      token
    )


    // Save the safe user profile.
    localStorage.setItem(
      'diapredict_user',
      JSON.stringify(user)
    )


    // Update React authentication state.
    setAuthToken(token)
    setCurrentUser(user)


    // Open the dashboard after login.
    setAuthView('login')
    setActivePage('dashboard')

  }


  // ========================================================
  // 6.8 LOGOUT HANDLER
  // ========================================================

  const handleLogout = () => {

    // Remove the saved JWT token.
    localStorage.removeItem('diapredict_access_token')

    // Remove the saved user profile.
    localStorage.removeItem('diapredict_user')


    // Clear authentication state.
    setAuthToken(null)
    setCurrentUser(null)


    // Clear history data from the previous session.
    setPredictionHistory([])
    setHistoryDashboard(null)
    setHistoryError('')


    // Return to the Login screen.
    setAuthView('login')
    setActivePage('dashboard')

  }

  // ========================================================
  // 6.8.1 REUSABLE AUTHENTICATED API REQUEST HANDLER
  // ========================================================

  const authenticatedFetch = async (url, options = {}) => {

    // Ensure the user has an authentication token.
    if (!authToken) {
      handleLogout()
      throw new Error('Authentication required. Please log in again.')
    }

    // Attach the JWT to the request automatically.
    const response = await fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: `Bearer ${authToken}`,
      },
    })

    // Automatically end the session if the backend returns HTTP 401.
    if (response.status === 401) {
      handleLogout()
      throw new Error('Your session has expired. Please log in again.')
    }

    // Preserve the original response for existing error handling.
    return response
  }


  // ========================================================
  // 6.9 AUTHENTICATION SCREEN ROUTING
  // ========================================================

  // Display Login or Register if authentication data
  // is not available.

  if (!authToken || !currentUser) {

    // Display the Registration component.
    if (authView === 'register') {

      return (
        <Register
          onGoToLogin={() => setAuthView('login')}
        />
      )

    }


    // Otherwise display the Login component.
    return (
      <Login
        onLoginSuccess={handleLoginSuccess}
        onGoToRegister={() => setAuthView('register')}
      />
    )

  }


  // ========================================================
  // 7. MAIN APPLICATION LAYOUT
  // ========================================================

  return (

    <div className="app-layout premium-layout">

      {/* ==================================================
          7.1 SIDEBAR
          ================================================== */}

      <aside className="sidebar premium-sidebar">

        {/* Application branding */}
        <div className="brand premium-brand">

          <div className="brand-icon premium-brand-icon">
            D
          </div>

          <div className="brand-text">

            <h2>
              DiaPredict
            </h2>

            <p>
              Intelligent Health Analytics
            </p>

          </div>

        </div>


        {/* Environment indicator */}
        <div className="sidebar-environment">

          <span className="environment-dot"></span>

          <span>
            AI RESEARCH PLATFORM
          </span>

        </div>


        {/* Sidebar navigation groups */}
        {['WORKSPACE', 'INTELLIGENCE', 'SYSTEM', 'ADMINISTRATION'].map((group) => (

          <div
            className="navigation-group"
            key={group}
          >

            <p className="sidebar-heading">
              {group}
            </p>


            <nav className="sidebar-navigation">

              {navigationItems

                .filter((item) => item.group === group)
                .filter((item) => !item.adminOnly || currentUser?.role === 'admin')

                .map((item) => (

                  <button
                    key={item.id}
                    type="button"
                    className={`nav-item premium-nav-item ${
                      activePage === item.id ? 'active' : ''
                    }`}
                    onClick={() => setActivePage(item.id)}
                    aria-current={
                      activePage === item.id ? 'page' : undefined
                    }
                  >

                    <span className="nav-icon">
                      {item.icon}
                    </span>

                    <span className="nav-label">
                      {item.label}
                    </span>

                    {activePage === item.id && (
                      <span className="nav-active-indicator"></span>
                    )}

                  </button>

                ))}

            </nav>

          </div>

        ))}


        {/* Sidebar footer */}
        <div className="sidebar-footer premium-sidebar-footer">

          <div className="sidebar-status">

            <span className="status-dot"></span>

            <span>
              Research Environment
            </span>

          </div>


          <div className="sidebar-project-info">

            <p>
              MAJOR PROJECT
            </p>

            <strong>
              DiaPredict AI
            </strong>

            <span>
              B.Tech Computer Science Engineering
            </span>

          </div>

        </div>

      </aside>


      {/* ==================================================
          7.2 MAIN CONTENT AREA
          ================================================== */}

      <main className="main-content premium-main-content">


        {/* ==================================================
            7.3 TOP HEADER
            ================================================== */}

        <header className="top-header premium-header">

          <div className="header-title">

            <div className="header-breadcrumb">

              <span>
                DIAPREDICT
              </span>

              <span className="breadcrumb-divider">
                /
              </span>

              <span>
                {pageTitle.toUpperCase()}
              </span>

            </div>

            <h1>
              {pageTitle}
            </h1>

          </div>


          {/* Header actions */}
          <div className="header-actions">

            <div className="header-system-status">

              <span className="status-dot"></span>

              <span>
                Research Mode
              </span>

            </div>


            {/* User profile */}
            <div className="header-profile">

              <div className="profile-avatar">

                {(currentUser.name || 'U')
                  .charAt(0)
                  .toUpperCase()}

              </div>

              <div className="profile-details">

                <p className="profile-name">
                  {currentUser.name}
                </p>

                <p className="profile-role">
                  {currentUser.role || 'User Account'}
                </p>

              </div>


              {/* Logout */}
              <button
                type="button"
                className="logout-button"
                onClick={handleLogout}
                aria-label="Log out"
              >
                Logout
              </button>

            </div>

          </div>

        </header>


        {/* ==================================================
            7.4 DASHBOARD PAGE
            ================================================== */}

        {activePage === 'dashboard' ? (

          <section className="dashboard-content premium-dashboard">

            {/* Welcome section */}
            <div className="premium-welcome">

              <div>

                <p className="welcome-eyebrow">
                  INTELLIGENT HEALTH ANALYTICS
                </p>

                <h2>
                  Welcome to <span>DiaPredict.</span>
                </h2>

                <p className="welcome-description">
                  Explore diabetes risk prediction, machine learning
                  performance and explainable AI insights in one place.
                </p>

              </div>


              <button
                type="button"
                className="primary-button premium-create-button"
                onClick={() => setActivePage('prediction')}
              >
                <span>＋</span>
                New Prediction
              </button>

            </div>


            {/* Dashboard statistics */}
            <section className="premium-statistics-section">

              <SectionHeading
                eyebrow="SYSTEM OVERVIEW"
                title="Performance at a glance"
                description="Research metrics and system information."
              />

              <div className="statistics-grid premium-statistics-grid">

                {dashboardStatistics.map((statistic) => (

                  <StatisticCard
                    key={statistic.title}
                    {...statistic}
                    value={
                      statistic.title === 'Total Predictions'
                        ? (historyDashboard?.total_predictions ?? '—')
                        : statistic.value
                    }
                  />

                ))}

              </div>

            </section>


            {/* Dashboard analytics grid */}
            <div className="premium-dashboard-grid">


              {/* Model performance */}
              <section className="dashboard-panel premium-panel performance-panel">

                <SectionHeading
                  eyebrow="MODEL EVALUATION"
                  title="Independent test performance"
                  description="Performance of the selected model on unseen test data."
                />


                <div className="performance-highlight">

                  <div>

                    <p>
                      Test Accuracy
                    </p>

                    <h2>
                      72.08<span>%</span>
                    </h2>

                    <small>
                      154 independent test records
                    </small>

                  </div>

                  <div className="performance-symbol">
                    ◎
                  </div>

                </div>


                {/* Independent test metrics */}
                <div className="performance-metrics">

                  <div className="performance-metric">
                    <span>Precision</span>
                    <strong>58.46%</strong>
                  </div>

                  <div className="performance-metric">
                    <span>Recall</span>
                    <strong>70.37%</strong>
                  </div>

                  <div className="performance-metric">
                    <span>F1 Score</span>
                    <strong>63.87%</strong>
                  </div>

                  <div className="performance-metric">
                    <span>ROC-AUC</span>
                    <strong>80.85%</strong>
                  </div>

                </div>


                <div className="research-note">

                  <span>ⓘ</span>

                  <p>
                    These are independent test-set results, not training
                    accuracy. The model has not been clinically validated.
                  </p>

                </div>

              </section>


              {/* Dataset information */}
              <section className="dashboard-panel premium-panel dataset-panel">

                <SectionHeading
                  eyebrow="DATASET OVERVIEW"
                  title="Pima Indians Diabetes Dataset"
                  description="Dataset characteristics used in model development."
                />


                <div className="dataset-total">

                  <div>

                    <p>
                      Total Records
                    </p>

                    <h2>
                      768
                    </h2>

                  </div>

                  <span className="dataset-icon">
                    ▤
                  </span>

                </div>


                <div className="dataset-distribution">

                  <div className="distribution-heading">

                    <span>
                      Outcome Distribution
                    </span>

                    <span>
                      Original dataset
                    </span>

                  </div>


                  <div className="distribution-bar">

                    <div
                      className="distribution-class-zero"
                      style={{ width: '65.1%' }}
                    ></div>

                    <div
                      className="distribution-class-one"
                      style={{ width: '34.9%' }}
                    ></div>

                  </div>


                  <div className="distribution-legend">

                    <div>
                      <span className="legend-dot class-zero"></span>
                      <span>Class 0</span>
                      <strong>500</strong>
                    </div>

                    <div>
                      <span className="legend-dot class-one"></span>
                      <span>Class 1</span>
                      <strong>268</strong>
                    </div>

                  </div>


                  <p className="dataset-footnote">
                    Class distribution is calculated from the original
                    dataset. Class 1 represents the positive outcome label.
                  </p>

                </div>

              </section>


              {/* Model information */}
              <section className="dashboard-panel premium-panel model-panel">

                <SectionHeading
                  eyebrow="MACHINE LEARNING"
                  title="Model configuration"
                  description="Current research model and processing pipeline."
                />


                <div className="model-information premium-model-information">

                  <div className="model-row">
                    <span>Algorithm</span>
                    <strong>Logistic Regression</strong>
                  </div>

                  <div className="model-row">
                    <span>Input Features</span>
                    <strong>8 Features</strong>
                  </div>

                  <div className="model-row">
                    <span>Training Strategy</span>
                    <strong>Stratified Split</strong>
                  </div>

                  <div className="model-row">
                    <span>Test Set</span>
                    <strong>20% · 154 Records</strong>
                  </div>

                  <div className="model-row">
                    <span>Explainability</span>
                    <strong>SHAP</strong>
                  </div>

                </div>


                <button
                  type="button"
                  className="secondary-button premium-panel-button"
                  onClick={() => setActivePage('insights')}
                >
                  Explore Model Insights →
                </button>

              </section>


              {/* Explainable AI */}
              <section className="dashboard-panel premium-panel explainability-panel">

                <SectionHeading
                  eyebrow="EXPLAINABLE ARTIFICIAL INTELLIGENCE"
                  title="Understand model decisions"
                  description="Explore the contribution of patient features to model predictions."
                />


                <div className="explainability-visual">

                  <div className="explainability-orbit orbit-one"></div>

                  <div className="explainability-orbit orbit-two"></div>

                  <div className="explainability-center">
                    <span>AI</span>
                  </div>

                  <span className="orbit-label orbit-label-one">
                    SHAP
                  </span>

                  <span className="orbit-label orbit-label-two">
                    FEATURES
                  </span>

                  <span className="orbit-label orbit-label-three">
                    MODEL
                  </span>

                </div>


                <button
                  type="button"
                  className="secondary-button premium-panel-button"
                  onClick={() => setActivePage('explainability')}
                >
                  Explore Explainable AI →
                </button>

              </section>

            </div>


            {/* Quick actions */}
            <section className="quick-actions-section">

              <SectionHeading
                eyebrow="QUICK ACCESS"
                title="Continue your research"
                description="Open a DiaPredict module."
              />


              <div className="quick-actions-grid">

                <button
                  type="button"
                  className="quick-action-card"
                  onClick={() => setActivePage('prediction')}
                >

                  <span className="quick-action-icon">
                    ✚
                  </span>

                  <span>
                    <strong>New Prediction</strong>
                    <small>Enter patient health information</small>
                  </span>

                  <span className="quick-action-arrow">→</span>

                </button>


                <button
                  type="button"
                  className="quick-action-card"
                  onClick={() => setActivePage('history')}
                >

                  <span className="quick-action-icon">
                    ◷
                  </span>

                  <span>
                    <strong>Prediction History</strong>
                    <small>Review previous predictions</small>
                  </span>

                  <span className="quick-action-arrow">→</span>

                </button>


                <button
                  type="button"
                  className="quick-action-card"
                  onClick={() => setActivePage('reports')}
                >

                  <span className="quick-action-icon">
                    ▤
                  </span>

                  <span>
                    <strong>Research Reports</strong>
                    <small>Explore project evaluation</small>
                  </span>

                  <span className="quick-action-arrow">→</span>

                </button>

              </div>

            </section>


            {/* Dashboard footer */}
            <footer className="dashboard-footer premium-footer">

              <p>
                DiaPredict AI · B.Tech CSE Major Project
              </p>

              <p>
                Explainable Machine Learning Research System
              </p>

            </footer>

          </section>


        ) : activePage === 'admin' && currentUser?.role === 'admin' ? (

          /* ==================================================
             ADMINISTRATION · USER MANAGEMENT
             Reads the protected /admin/users endpoint.
             ================================================== */

          <section className="dashboard-content history-page">
            <div className="history-page-heading">
              <div>
                <p className="section-eyebrow">ADMINISTRATION</p>
                <h2>User Management</h2>
                <p>Review registered DiaPredict accounts and their assigned roles.</p>
              </div>
              <button
                type="button"
                className="secondary-button"
                onClick={() => {
                  setAdminError('')
                  setActivePage('dashboard')
                  setTimeout(() => setActivePage('admin'), 0)
                }}
              >
                Refresh Users ↻
              </button>
            </div>

            {adminDashboard?.statistics && (
              <div className="history-summary-grid" style={{ marginBottom: '22px' }}>
                {[
                  ['Registered Users', adminDashboard.statistics.total_users ?? 0],
                  ['Total Predictions', adminDashboard.statistics.total_predictions ?? 0],
                  ['Class 0 Predictions', adminDashboard.statistics.class_0_predictions ?? 0],
                  ['Class 1 Predictions', adminDashboard.statistics.class_1_predictions ?? 0],
                ].map(([label, value]) => (
                  <article className="history-summary-card" key={label}>
                    <span>{label}</span>
                    <strong>{value}</strong>
                    <small>System-wide aggregate</small>
                  </article>
                ))}
              </div>
            )}

            <section className="dashboard-panel premium-panel history-table-panel">
              <div className="history-table-heading">
                <div>
                  <h3>Registered Accounts</h3>
                  <p>Only account ID, name, email and role are displayed. Password hashes are never returned.</p>
                </div>
                <span className="history-live-tag">
                  {adminLoading ? 'LOADING USERS' : adminError ? 'API ERROR' : 'ADMIN VIEW'}
                </span>
              </div>

              {adminError && (
                <div className="history-empty-state" role="alert">
                  <h3>Unable to load User Management</h3>
                  <p>{adminError}</p>
                </div>
              )}

              {adminLoading && (
                <div className="history-empty-state">
                  <h3>Loading registered users...</h3>
                  <p>Retrieving safe account information from the database.</p>
                </div>
              )}

              {!adminLoading && !adminError && (
                <>
                  <div className="history-controls">
                    <input
                      type="search"
                      className="history-search"
                      placeholder="Search by user ID, name, email or role..."
                      aria-label="Search registered users"
                      value={adminUserSearch}
                      onChange={(event) => setAdminUserSearch(event.target.value)}
                    />
                    <span className="history-live-tag">
                      {filteredAdminUsers.length} USERS
                    </span>
                  </div>

                  {filteredAdminUsers.length === 0 ? (
                    <div className="history-empty-state">
                      <h3>No matching users</h3>
                      <p>Try another name, email address, user ID or role.</p>
                    </div>
                  ) : (
                    <div className="history-table-wrapper">
                      <table className="history-table">
                        <thead>
                          <tr>
                            <th>User ID</th>
                            <th>Name</th>
                            <th>Email</th>
                            <th>Role</th>
                          </tr>
                        </thead>
                        <tbody>
                          {filteredAdminUsers.map((user) => (
                            <tr key={user.id}>
                              <td>#{user.id}</td>
                              <td>{user.name}</td>
                              <td>{user.email}</td>
                              <td>
                                <span className={user.role === 'admin' ? 'history-result-positive' : 'history-result-negative'}>
                                  {user.role}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}
            </section>
          </section>

        ) : activePage === 'analytics' ? (

          /* ==================================================
             7.5 ADVANCED ANALYTICS PAGE
             Live analytics for the authenticated user.
             ================================================== */

          <section className="dashboard-content history-page">
            <div className="history-page-heading">
              <div>
                <h2>Advanced Analytics</h2>
                <p>
                  Explore trends, prediction outcomes, confidence levels
                  and input patterns from your prediction history.
                </p>
              </div>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setActivePage('history')}
              >
                View Prediction History →
              </button>
            </div>

            {analyticsLoading && (
              <div className="dashboard-panel premium-panel" style={{ padding: '28px', marginBottom: '20px' }}>
                <h3>Loading advanced analytics...</h3>
                <p style={{ opacity: 0.75, marginBottom: 0 }}>
                  Retrieving your prediction data from the DiaPredict database.
                </p>
              </div>
            )}

            {!analyticsLoading && analyticsError && (
              <div className="dashboard-panel premium-panel" role="alert" style={{ padding: '24px', marginBottom: '20px' }}>
                <h3>Unable to load analytics</h3>
                <p style={{ color: '#ffb4b4' }}>{analyticsError}</p>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    setAnalyticsError('')
                    setActivePage('dashboard')
                    setTimeout(() => setActivePage('analytics'), 0)
                  }}
                >
                  Try Again
                </button>
              </div>
            )}

            {!analyticsLoading && !analyticsError && analyticsData && (
              <>
                <div className="history-summary-grid" style={{ marginBottom: '22px' }}>
                  {[
                    {
                      label: 'Total Predictions',
                      value: analyticsData.summary?.total_predictions ?? 0,
                      note: 'Your recorded assessments',
                    },
                    {
                      label: 'Class 0 Predictions',
                      value: analyticsData.summary?.class_0_predictions ?? 0,
                      note: 'Negative outcome label',
                    },
                    {
                      label: 'Class 1 Predictions',
                      value: analyticsData.summary?.class_1_predictions ?? 0,
                      note: 'Positive outcome label',
                    },
                    {
                      label: 'Average Confidence',
                      value: `${Number(analyticsData.summary?.average_confidence ?? 0).toFixed(2)}%`,
                      note: 'Predicted-class confidence',
                    },
                  ].map((item) => (
                    <article className="history-summary-card" key={item.label}>
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                      <small>{item.note}</small>
                    </article>
                  ))}
                </div>

                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 360px), 1fr))',
                    gap: '20px',
                    marginBottom: '20px',
                  }}
                >
                  <section className="dashboard-panel premium-panel" style={{ padding: '24px' }}>
                    <p className="section-eyebrow">OUTCOME ANALYTICS</p>
                    <h3 style={{ margin: '4px 0 8px' }}>Prediction Distribution</h3>
                    <p style={{ opacity: 0.72, fontSize: '13px' }}>
                      Class 0 and Class 1 predictions in your account.
                    </p>

                    {(analyticsData.outcome_distribution || []).map((item) => {
                      const color = item.label === 'Class 1' ? '#A78BFA' : '#20D3E8'
                      const percentage = Math.max(0, Math.min(100, Number(item.percentage) || 0))

                      return (
                        <div key={item.label} style={{ marginTop: '20px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px', marginBottom: '8px' }}>
                            <strong>{item.label}</strong>
                            <span>{item.count} · {percentage.toFixed(1)}%</span>
                          </div>
                          <div style={{ height: '11px', borderRadius: '20px', background: 'rgba(148, 163, 184, 0.16)', overflow: 'hidden' }}>
                            <div style={{ width: `${percentage}%`, height: '100%', borderRadius: '20px', background: color }} />
                          </div>
                        </div>
                      )
                    })}
                  </section>

                  <section className="dashboard-panel premium-panel" style={{ padding: '24px' }}>
                    <p className="section-eyebrow">PROBABILITY ANALYSIS</p>
                    <h3 style={{ margin: '4px 0 8px' }}>Confidence Distribution</h3>
                    <p style={{ opacity: 0.72, fontSize: '13px' }}>
                      Confidence ranges for the predicted class.
                    </p>

                    {(analyticsData.probability_distribution || []).map((item) => {
                      const maximum = Math.max(
                        1,
                        ...(analyticsData.probability_distribution || []).map((entry) => Number(entry.count) || 0)
                      )
                      const count = Number(item.count) || 0

                      return (
                        <div key={item.range} style={{ display: 'grid', gridTemplateColumns: '76px 1fr 28px', alignItems: 'center', gap: '10px', marginTop: '15px' }}>
                          <span style={{ fontSize: '12px', opacity: 0.8 }}>{item.range}</span>
                          <div style={{ height: '9px', borderRadius: '20px', background: 'rgba(148, 163, 184, 0.16)', overflow: 'hidden' }}>
                            <div style={{ width: `${(count / maximum) * 100}%`, height: '100%', borderRadius: '20px', background: '#5599FF' }} />
                          </div>
                          <strong style={{ fontSize: '13px', textAlign: 'right' }}>{count}</strong>
                        </div>
                      )
                    })}
                    <p style={{ marginTop: '18px', fontSize: '12px', opacity: 0.65 }}>
                      Confidence is not the same as clinical certainty or diagnostic accuracy.
                    </p>
                  </section>
                </div>

                <section className="dashboard-panel premium-panel" style={{ padding: '24px', marginBottom: '20px' }}>
                  <p className="section-eyebrow">PREDICTION ACTIVITY</p>
                  <h3 style={{ margin: '4px 0 8px' }}>Daily Prediction Trends</h3>
                  <p style={{ opacity: 0.72, fontSize: '13px' }}>
                    Number of predictions recorded per day during the available trend period.
                  </p>

                  {(analyticsData.daily_trends || []).length === 0 ? (
                    <p style={{ opacity: 0.7 }}>No daily trend records are available yet.</p>
                  ) : (
                    <div style={{ display: 'grid', gap: '15px', marginTop: '20px' }}>
                      {(analyticsData.daily_trends || []).map((day) => {
                        const total = Number(day.total) || 0
                        const classZero = Number(day.class_0) || 0
                        const classOne = Number(day.class_1) || 0
                        const maxTotal = Math.max(
                          1,
                          ...(analyticsData.daily_trends || []).map((entry) => Number(entry.total) || 0)
                        )

                        return (
                          <div key={day.date} style={{ display: 'grid', gridTemplateColumns: '100px minmax(80px, 1fr) 42px', alignItems: 'center', gap: '12px' }}>
                            <span style={{ fontSize: '12px', opacity: 0.8 }}>{day.date}</span>
                            <div style={{ height: '13px', display: 'flex', borderRadius: '20px', overflow: 'hidden', background: 'rgba(148, 163, 184, 0.16)' }}>
                              <div title={`Class 0: ${classZero}`} style={{ width: `${(classZero / maxTotal) * 100}%`, background: '#20D3E8' }} />
                              <div title={`Class 1: ${classOne}`} style={{ width: `${(classOne / maxTotal) * 100}%`, background: '#A78BFA' }} />
                            </div>
                            <strong style={{ textAlign: 'right' }}>{total}</strong>
                          </div>
                        )
                      })}
                    </div>
                  )}

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '18px', marginTop: '20px', fontSize: '12px' }}>
                    <span><span style={{ display: 'inline-block', width: '9px', height: '9px', borderRadius: '50%', background: '#20D3E8', marginRight: '7px' }} />Class 0</span>
                    <span><span style={{ display: 'inline-block', width: '9px', height: '9px', borderRadius: '50%', background: '#A78BFA', marginRight: '7px' }} />Class 1</span>
                  </div>
                </section>

                <section className="dashboard-panel premium-panel" style={{ padding: '24px', marginBottom: '20px' }}>
                  <p className="section-eyebrow">PATIENT INPUT ANALYTICS</p>
                  <h3 style={{ margin: '4px 0 8px' }}>Input Averages and Ranges</h3>
                  <p style={{ opacity: 0.72, fontSize: '13px' }}>
                    Descriptive statistics calculated from your submitted prediction inputs.
                  </p>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '14px', marginTop: '20px' }}>
                    {[
                      {
                        label: 'Average Glucose',
                        average: analyticsData.input_averages?.glucose,
                        minimum: analyticsData.input_ranges?.glucose?.minimum,
                        maximum: analyticsData.input_ranges?.glucose?.maximum,
                        unit: 'mg/dL',
                      },
                      {
                        label: 'Average BMI',
                        average: analyticsData.input_averages?.bmi,
                        minimum: analyticsData.input_ranges?.bmi?.minimum,
                        maximum: analyticsData.input_ranges?.bmi?.maximum,
                        unit: 'kg/m²',
                      },
                      {
                        label: 'Average Age',
                        average: analyticsData.input_averages?.age,
                        unit: 'years',
                      },
                    ].map((item) => (
                      <article key={item.label} className="history-summary-card">
                        <span>{item.label}</span>
                        <strong>
                          {item.average === null || item.average === undefined
                            ? '—'
                            : Number(item.average).toFixed(1)}
                        </strong>
                        <small>{item.unit}</small>
                        {item.minimum !== undefined && item.maximum !== undefined && (
                          <small style={{ display: 'block', marginTop: '7px' }}>
                            Range: {item.minimum} – {item.maximum}
                          </small>
                        )}
                      </article>
                    ))}
                  </div>
                </section>

                <section className="dashboard-panel premium-panel" style={{ padding: '24px' }}>
                  <p className="section-eyebrow">ANALYTICS SUMMARY</p>
                  <h3 style={{ margin: '4px 0 8px' }}>Your Prediction Activity</h3>
                  <p style={{ lineHeight: 1.7, opacity: 0.82, marginBottom: 0 }}>
                    Your account contains <strong>{analyticsData.summary?.total_predictions ?? 0}</strong> prediction records.
                    There are <strong>{analyticsData.summary?.class_0_predictions ?? 0}</strong> Class 0 and{' '}
                    <strong>{analyticsData.summary?.class_1_predictions ?? 0}</strong> Class 1 outcomes.
                    The average predicted-class confidence is{' '}
                    <strong>{Number(analyticsData.summary?.average_confidence ?? 0).toFixed(2)}%</strong>.
                    These figures describe your stored model outputs only; they do not establish a medical diagnosis
                    or the clinical accuracy of the model.
                  </p>
                </section>
              </>
            )}

            {!analyticsLoading && !analyticsError && !analyticsData && (
              <div className="dashboard-panel premium-panel" style={{ padding: '24px' }}>
                <h3>No analytics data available</h3>
                <p style={{ opacity: 0.75 }}>
                  Complete a prediction to begin building your analytics history.
                </p>
              </div>
            )}
          </section>


        ) : activePage === 'prediction' ? (

          /* ==================================================
             7.5 PREDICTION PAGE
             Pass the authenticated user's JWT token.
             Prediction.jsx receives it through props.
             ================================================== */

          <Prediction
              authToken={authToken}
              onLogout={handleLogout}
          />


        ) : activePage === 'history' ? (

          /* ==================================================
             7.6 PREDICTION HISTORY PAGE
             Displays real records returned by FastAPI.
             ================================================== */

          <section className="dashboard-content history-page">


            {/* History page heading */}
            <div className="history-page-heading">

              <div>

                <h2>
                  Prediction History
                </h2>

                <p>
                  Review your previous diabetes risk assessments.
                </p>

              </div>


              <button
                type="button"
                className="predict-button"
                onClick={() => setActivePage('prediction')}
              >
                + New Prediction
              </button>

            </div>


            {/* ==================================================
                HISTORY SUMMARY CARDS
                Values come from the dashboard API.
                ================================================== */}

            <div className="history-summary-grid">


              <article className="history-summary-card">

                <span>
                  Total Records
                </span>

                <strong>
                  {historyDashboard?.total_predictions ?? '—'}
                </strong>

                <small>
                  Predictions in your account
                </small>

              </article>


              <article className="history-summary-card">

                <span>
                  Positive Predictions
                </span>

                <strong>
                  {historyDashboard?.class_1_predictions ?? '—'}
                </strong>

                <small>
                  Positive outcome predictions
                </small>

              </article>


              <article className="history-summary-card">

                <span>
                  Negative Predictions
                </span>

                <strong>
                  {historyDashboard?.class_0_predictions ?? '—'}
                </strong>

                <small>
                  Negative outcome predictions
                </small>

              </article>

            </div>


            {/* ==================================================
                PREVIOUS ASSESSMENTS PANEL
                ================================================== */}

            <div className="dashboard-panel premium-panel history-table-panel">


              <div className="history-table-heading">

                <div>

                  <h3>
                    Previous Assessments
                  </h3>

                  <p>
                    Your authenticated prediction records.
                  </p>

                </div>


                <span className="history-live-tag">

                  {historyLoading
                    ? 'LOADING DATA'
                    : historyError
                      ? 'API ERROR'
                      : 'DATABASE CONNECTED'}

                </span>

              </div>


              {/* ==================================================
                  SEARCH AND FILTER CONTROLS
                  ================================================== */}

              <div className="history-controls">


                {/* Search by ID, age, glucose, BMI or date */}
                <input
                  type="search"
                  className="history-search"
                  placeholder="Search by ID, age, glucose, BMI or date..."
                  aria-label="Search prediction records"
                  value={historySearch}
                  onChange={(event) => setHistorySearch(event.target.value)}
                />


                {/* Filter by predicted outcome */}
                <select
                  className="history-filter"
                  aria-label="Filter prediction records"
                  value={historyFilter}
                  onChange={(event) => setHistoryFilter(event.target.value)}
                >

                  <option value="all">
                    All Predictions
                  </option>

                  <option value="positive">
                    Positive
                  </option>

                  <option value="negative">
                    Negative
                  </option>

                </select>

                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleDownloadPredictionHistoryCsv}
                  disabled={downloadingCsv || historyLoading || predictionHistory.length === 0}
                  aria-label="Download prediction history as CSV"
                  style={{
                    padding: '10px 14px',
                    whiteSpace: 'nowrap',
                    opacity: downloadingCsv ? 0.65 : 1,
                    cursor: downloadingCsv ? 'wait' : 'pointer',
                  }}
                >
                  {downloadingCsv ? 'Preparing CSV...' : '↓ Download CSV'}
                </button>

              </div>

              {csvDownloadError && (
                <div
                  className="history-empty-state"
                  role="alert"
                  style={{ marginTop: '14px', color: '#ffb4b4' }}
                >
                  <p>{csvDownloadError}</p>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setCsvDownloadError('')}
                  >
                    Dismiss
                  </button>
                </div>
              )}


              {/* ==================================================
                  LOADING STATE
                  ================================================== */}

              {historyLoading && (

                <div className="history-empty-state">

                  <div className="history-empty-icon">
                    ◌
                  </div>

                  <h3>
                    Loading prediction history
                  </h3>

                  <p>
                    Connecting to the DiaPredict database...
                  </p>

                </div>

              )}


              {/* ==================================================
                  API ERROR STATE
                  ================================================== */}

              {!historyLoading && historyError && (

                <div className="history-empty-state">

                  <div className="history-empty-icon">
                    !
                  </div>

                  <h3>
                    Unable to load prediction history
                  </h3>

                  <p>
                    {historyError}
                  </p>

                  <button
                    type="button"
                    className="predict-button"
                    onClick={() => {
                      setHistoryError('')
                      setActivePage('dashboard')
                      setTimeout(() => setActivePage('history'), 0)
                    }}
                  >
                    Try Again
                  </button>

                </div>

              )}


              {/* ==================================================
                  EMPTY HISTORY STATE
                  ================================================== */}

              {!historyLoading &&
                !historyError &&
                predictionHistory.length === 0 && (

                  <div className="history-empty-state">

                    <div className="history-empty-icon">
                      ◷
                    </div>

                    <h3>
                      No prediction records found
                    </h3>

                    <p>
                      Your previous predictions will appear here
                      after you complete an assessment.
                    </p>

                    <button
                      type="button"
                      className="predict-button"
                      onClick={() => setActivePage('prediction')}
                    >
                      Create Your First Prediction →
                    </button>

                  </div>

                )}


              {/* ==================================================
                  NO SEARCH RESULTS STATE
                  ================================================== */}

              {!historyLoading &&
                !historyError &&
                predictionHistory.length > 0 &&
                filteredHistory.length === 0 && (

                  <div className="history-empty-state">

                    <div className="history-empty-icon">
                      ⌕
                    </div>

                    <h3>
                      No matching records
                    </h3>

                    <p>
                      Try changing your search text or prediction filter.
                    </p>

                  </div>

                )}


              {/* ==================================================
                  SELECTED PREDICTION DETAILS
                  ================================================== */}

              {selectedHistoryRecord && (
                <section
                  className="dashboard-panel premium-panel"
                  aria-label={`Details for prediction ${selectedHistoryRecord.id}`}
                  style={{ marginTop: '22px', padding: '24px' }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      justifyContent: 'space-between',
                      gap: '16px',
                      flexWrap: 'wrap',
                      marginBottom: '20px',
                    }}
                  >
                    <div>
                      <p className="section-eyebrow">PREDICTION RECORD</p>
                      <h3 style={{ margin: '4px 0 6px' }}>
                        Prediction Details · #{selectedHistoryRecord.id}
                      </h3>
                      <p style={{ margin: 0, opacity: 0.75 }}>
                        {selectedHistoryRecord.created_at
                          ? new Date(selectedHistoryRecord.created_at).toLocaleString()
                          : 'Date unavailable'}
                      </p>
                    </div>

                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => setSelectedHistoryPredictionId(null)}
                    >
                      Close Details ×
                    </button>
                  </div>

                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                      gap: '14px',
                      marginBottom: '22px',
                    }}
                  >
                    <article className="history-summary-card">
                      <span>Predicted Outcome</span>
                      <strong>
                        {Number(selectedHistoryRecord.predicted_class) === 1
                          ? 'Positive'
                          : 'Negative'}
                      </strong>
                      <small>Model classification</small>
                    </article>

                    <article className="history-summary-card">
                      <span>Class 1 Probability</span>
                      <strong>{(selectedClass1Probability * 100).toFixed(2)}%</strong>
                      <small>Positive class probability</small>
                    </article>

                    <article className="history-summary-card">
                      <span>Class 0 Probability</span>
                      <strong>{(selectedClass0Probability * 100).toFixed(2)}%</strong>
                      <small>Negative class probability</small>
                    </article>
                  </div>

                  <h4 style={{ margin: '0 0 14px' }}>Submitted Input Parameters</h4>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                      gap: '12px',
                      marginBottom: '26px',
                    }}
                  >
                    {[
                      ['Pregnancies', selectedHistoryRecord.pregnancies],
                      ['Glucose', selectedHistoryRecord.glucose],
                      ['Blood Pressure', selectedHistoryRecord.blood_pressure],
                      ['Skin Thickness', selectedHistoryRecord.skin_thickness],
                      ['Insulin', selectedHistoryRecord.insulin],
                      ['BMI', selectedHistoryRecord.bmi],
                      ['Diabetes Pedigree Function', selectedHistoryRecord.diabetes_pedigree_function],
                      ['Age', selectedHistoryRecord.age],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        style={{
                          padding: '14px 16px',
                          border: '1px solid rgba(148, 163, 184, 0.2)',
                          borderRadius: '12px',
                          background: 'rgba(15, 23, 42, 0.24)',
                        }}
                      >
                        <div style={{ fontSize: '12px', opacity: 0.72, marginBottom: '6px' }}>
                          {label}
                        </div>
                        <strong style={{ fontSize: '17px' }}>
                          {value ?? '—'}
                        </strong>
                      </div>
                    ))}
                  </div>

                  <div style={{ marginBottom: '14px' }}>
                    <h4 style={{ margin: '0 0 6px' }}>Individual SHAP Explanation</h4>
                    <p style={{ margin: 0, opacity: 0.75, fontSize: '13px' }}>
                      The waterfall below explains how each input feature contributes
                      to this prediction. SHAP values are shown in model-output
                      (log-odds) space, not probability.
                    </p>
                  </div>

                  <ShapWaterfall
                    key={`history-shap-${selectedHistoryRecord.id}`}
                    authToken={authToken}
                    initialPredictionId={String(selectedHistoryRecord.id)}
                    lockSelection
                  />

                  <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '18px' }}>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => handleDownloadPredictionReport(selectedHistoryRecord.id)}
                      disabled={downloadingReportId === selectedHistoryRecord.id}
                    >
                      {downloadingReportId === selectedHistoryRecord.id
                        ? 'Preparing PDF...'
                        : '↓ Download This Prediction PDF'}
                    </button>
                  </div>
                </section>
              )}


              {/* ==================================================
                  PREDICTION HISTORY TABLE
                  ================================================== */}

              {!historyLoading &&
                !historyError &&
                filteredHistory.length > 0 && (

                  <div className="history-table-wrapper">

                    <table className="history-table">

                      <thead>

                        <tr>
                          <th>Record ID</th>
                          <th>Date</th>
                          <th>Age</th>
                          <th>Glucose</th>
                          <th>BMI</th>
                          <th>Prediction</th>
                          <th>Probability</th>
                          <th>Details</th>
                          <th>Report</th>
                        </tr>

                      </thead>


                      <tbody>

                        {filteredHistory.map((record) => (

                          <tr key={record.id}>

                            <td>
                              #{record.id}
                            </td>

                            <td>
                              {record.created_at
                                ? new Date(record.created_at).toLocaleString()
                                : '—'}
                            </td>

                            <td>
                              {record.age}
                            </td>

                            <td>
                              {record.glucose}
                            </td>

                            <td>
                              {record.bmi}
                            </td>

                            <td>

                              <span
                                className={
                                  record.predicted_class === 1
                                    ? 'history-result-positive'
                                    : 'history-result-negative'
                                }
                              >

                                {record.predicted_class === 1
                                  ? 'Positive'
                                  : 'Negative'}

                              </span>

                            </td>

                            <td>

                              {(
                                Number(record.predicted_probability) * 100
                              ).toFixed(1)}%

                            </td>

                            <td>
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={() => setSelectedHistoryPredictionId(record.id)}
                                aria-label={`View details for prediction ${record.id}`}
                                style={{
                                  padding: '8px 12px',
                                  fontSize: '12px',
                                  whiteSpace: 'nowrap',
                                }}
                              >
                                View Details →
                              </button>
                            </td>

                            <td>
                              <button
                                type="button"
                                className="secondary-button"
                                onClick={() => handleDownloadPredictionReport(record.id)}
                                disabled={downloadingReportId === record.id}
                                aria-label={`Download PDF report for prediction ${record.id}`}
                                style={{
                                  padding: '8px 12px',
                                  fontSize: '12px',
                                  whiteSpace: 'nowrap',
                                  opacity: downloadingReportId === record.id ? 0.65 : 1,
                                  cursor: downloadingReportId === record.id ? 'wait' : 'pointer',
                                }}
                              >
                                {downloadingReportId === record.id
                                  ? 'Preparing...'
                                  : '↓ Download PDF'}
                              </button>
                            </td>

                          </tr>

                        ))}

                      </tbody>

                    </table>

                  </div>

                )}


              {reportDownloadError && (
                <div
                  className="history-empty-state"
                  role="alert"
                  style={{ marginTop: '14px', color: '#ffb4b4' }}
                >
                  <p>{reportDownloadError}</p>
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setReportDownloadError('')}
                  >
                    Dismiss
                  </button>
                </div>
              )}


              {/* Academic disclaimer */}
              <p className="history-disclaimer">
                Prediction results are generated by a research model
                and must not be interpreted as a medical diagnosis.
              </p>

            </div>

          </section>

        ) : activePage === 'explainability' ? (
          /* ==================================================
             7.7 EXPLAINABLE AI PAGE
             Displays one SHAP visualization at a time.
             ================================================== */

          <section className="dashboard-content explainability-page">

            {/* Explainable AI page heading */}
            <div className="history-page-heading">

              <div>

                {/* Main page title */}
                <h2>
                  Explainable AI
                </h2>

                {/* Brief explanation of this module */}
                <p>
                  Understand how machine-learning features influence model predictions.
                </p>

              </div>

            </div>


            {/* ==================================================
                SHAP VISUALIZATION TAB NAVIGATION
                Allows users to switch between four graphs.
                ================================================== */}

            <div
              className="shap-tabs"
              role="tablist"
              aria-label="SHAP visualization tabs"
            >

              {/* Tab 1: Global Feature Importance */}
              <button
                type="button"
                role="tab"
                aria-selected={activeShapTab === 'global'}
                className={`shap-tab ${
                  activeShapTab === 'global' ? 'active' : ''
                }`}
                onClick={() => setActiveShapTab('global')}
              >
                <span className="shap-tab-icon">▥</span>
                <span>Global Importance</span>
              </button>


              {/* Tab 2: SHAP Beeswarm */}
              <button
                type="button"
                role="tab"
                aria-selected={activeShapTab === 'beeswarm'}
                className={`shap-tab ${
                  activeShapTab === 'beeswarm' ? 'active' : ''
                }`}
                onClick={() => setActiveShapTab('beeswarm')}
              >
                <span className="shap-tab-icon">✣</span>
                <span>Beeswarm Plot</span>
              </button>


              {/* Tab 3: SHAP Dependence */}
              <button
                type="button"
                role="tab"
                aria-selected={activeShapTab === 'dependence'}
                className={`shap-tab ${
                  activeShapTab === 'dependence' ? 'active' : ''
                }`}
                onClick={() => setActiveShapTab('dependence')}
              >
                <span className="shap-tab-icon">⌁</span>
                <span>Dependence Plot</span>
              </button>


              {/* Tab 4: Individual SHAP Waterfall */}
              <button
                type="button"
                role="tab"
                aria-selected={activeShapTab === 'waterfall'}
                className={`shap-tab ${
                  activeShapTab === 'waterfall' ? 'active' : ''
                }`}
                onClick={() => setActiveShapTab('waterfall')}
              >
                <span className="shap-tab-icon">⌁</span>
                <span>Waterfall Plot</span>
              </button>

            </div>



            {/* ==================================================
                DYNAMIC SHAP VISUALIZATION DESCRIPTION
                Changes automatically when a tab is selected.
                ================================================== */}

            <div className="shap-info-panel">

              {/* Dynamic heading based on selected tab */}
              <h3>
                {activeShapTab === 'global' && 'Global Feature Importance'}

                {activeShapTab === 'beeswarm' && 'SHAP Beeswarm Analysis'}

                {activeShapTab === 'dependence' && 'SHAP Dependence Analysis'}

                {activeShapTab === 'waterfall' && 'Individual Prediction Explanation'}
              </h3>

              {/* Dynamic description based on selected tab */}
              <p>
                {activeShapTab === 'global' &&
                  'Understand which features have the greatest average absolute influence on the model output across the analysed dataset.'}

                {activeShapTab === 'beeswarm' &&
                  'Explore how individual feature values influence model predictions across multiple observations, including the direction and distribution of SHAP contributions.'}

                {activeShapTab === 'dependence' &&
                  'Explore the relationship between a feature value and its SHAP contribution to understand how changes in that feature influence model output.'}

                {activeShapTab === 'waterfall' &&
                  'Understand how individual patient features move the model output from its baseline value to the final prediction for a selected record.'}
              </p>

            </div>


            {/* ==================================================
                SELECTED SHAP VISUALIZATION
                Only the selected component is rendered.
                ================================================== */}

            <div className="shap-tab-content">

              {/* Display Global Feature Importance */}
              {activeShapTab === 'global' && (
                <GlobalShapImportance authToken={authToken} />
              )}

              {/* Display SHAP Beeswarm */}
              {activeShapTab === 'beeswarm' && (
                <ShapBeeswarm
                  authToken={authToken}
                  onLogout={handleLogout}
                />
              )}

              {/* Display SHAP Dependence Plot */}
              {activeShapTab === 'dependence' && (
                <ShapDependence authToken={authToken} />
              )}

              {/* Display Individual SHAP Waterfall */}
              {activeShapTab === 'waterfall' && (
                <ShapWaterfall authToken={authToken} />
              )}

            </div>


            {/* Academic disclaimer */}
            <p className="history-disclaimer">
              SHAP visualizations explain the machine-learning model's
              behaviour. They do not establish medical causation or
              replace professional medical diagnosis.
            </p>

          </section>


        ) : activePage === 'insights' ? (

          /* ==================================================
             MODEL INSIGHTS PAGE
             Reads authenticated dataset and research metrics
             from the dedicated FastAPI insights endpoint.
             ================================================== */

          <ModelInsights
            authToken={authToken}
            onLogout={handleLogout}
          />


        ) : activePage === 'reports' ? (

          /* ==================================================
             7.8 RESEARCH REPORTS PAGE
             Displays actual Phase 10 research evaluation.
             All metrics are from the independent test set.
             ================================================== */

          <section className="dashboard-content research-reports-page">

            {/* ==================================================
                RESEARCH REPORT PAGE HEADING
                ================================================== */}

            <div className="history-page-heading">

              <div>

                {/* Main page title */}
                <h2>
                  Research Reports
                </h2>

                {/* Page description */}
                <p>
                  Explore model evaluation, experimental results
                  and research findings.
                </p>

              </div>

              <div className="research-report-actions">
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleDownloadResearchReport}
                  disabled={downloadingResearchReport}
                >
                  {downloadingResearchReport ? 'Generating PDF...' : '↓ Download Research Report PDF'}
                </button>
              </div>

            </div>

            {researchReportDownloadError && (
              <p className="history-error" role="alert">
                {researchReportDownloadError}
              </p>
            )}

            {/* Research report tabs */}
            <div className="shap-tabs report-tabs" role="tablist" aria-label="Research report sections">
              {[
                ['overview', '1', 'Overview'],
                ['confusion', '2', 'Confusion Matrix'],
                ['methodology', '3', 'Dataset & Methodology'],
                ['findings', '4', 'Research Findings'],
                ['limitations', '5', 'Limitations'],
              ].map(([id, number, label]) => (
                <button key={id} type="button" role="tab"
                  aria-selected={activeReportTab === id}
                  className={`shap-tab ${activeReportTab === id ? 'active' : ''}`}
                  onClick={() => setActiveReportTab(id)}>
                  <span className="shap-tab-icon">{number}</span><span>{label}</span>
                </button>
              ))}
            </div>

            {/* ==================================================
                RESEARCH EVALUATION OVERVIEW
                Metrics from the independent test dataset.
                ================================================== */}

            {activeReportTab === 'overview' && (
            <section className="research-report-section">

              <div className="research-report-heading">

                <div>

                  <p className="section-eyebrow">
                    PHASE 10 · MODEL EVALUATION
                  </p>

                  <h3>
                    Independent Test Performance
                  </h3>

                  <p>
                    Evaluation results obtained from 154 unseen
                    test records.
                  </p>

                </div>

                <span className="research-report-badge">
                  RESEARCH RESULTS
                </span>

              </div>


              {/* Performance metric cards */}
              <div className="research-metrics-grid">

                {/* Accuracy */}
                <article className="research-metric-card">

                  <span className="research-metric-label">
                    Accuracy
                  </span>

                  <strong>
                    72.08%
                  </strong>

                  <small>
                    Overall correct predictions
                  </small>

                </article>


                {/* Precision */}
                <article className="research-metric-card">

                  <span className="research-metric-label">
                    Precision
                  </span>

                  <strong>
                    58.46%
                  </strong>

                  <small>
                    Positive prediction precision
                  </small>

                </article>


                {/* Recall */}
                <article className="research-metric-card">

                  <span className="research-metric-label">
                    Recall
                  </span>

                  <strong>
                    70.37%
                  </strong>

                  <small>
                    Positive cases identified
                  </small>

                </article>


                {/* Specificity */}
                <article className="research-metric-card">

                  <span className="research-metric-label">
                    Specificity
                  </span>

                  <strong>
                    73.00%
                  </strong>

                  <small>
                    Negative cases identified
                  </small>

                </article>


                {/* F1 Score */}
                <article className="research-metric-card">

                  <span className="research-metric-label">
                    F1 Score
                  </span>

                  <strong>
                    63.87%
                  </strong>

                  <small>
                    Harmonic mean of precision and recall
                  </small>

                </article>


                {/* ROC-AUC */}
                <article className="research-metric-card">

                  <span className="research-metric-label">
                    ROC-AUC
                  </span>

                  <strong>
                    80.85%
                  </strong>

                  <small>
                    Area under ROC curve
                  </small>

                </article>

              </div>

            </section>
            )}



            {/* ==================================================
                CONFUSION MATRIX
                Values from the independent test evaluation.
                ================================================== */}

            {activeReportTab === 'confusion' && (
            <section className="dashboard-panel premium-panel research-confusion-panel">

              <div className="research-report-heading">

                <div>

                  <p className="section-eyebrow">
                    CLASSIFICATION ANALYSIS
                  </p>

                  <h3>
                    Confusion Matrix
                  </h3>

                  <p>
                    Comparison of actual and predicted class labels.
                  </p>

                </div>

              </div>


              <div className="confusion-matrix-wrapper">

                <table className="research-confusion-matrix">

                  <thead>

                    <tr>

                      <th>
                        Actual / Predicted
                      </th>

                      <th>
                        Predicted Negative
                      </th>

                      <th>
                        Predicted Positive
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {/* Actual negative class */}
                    <tr>

                      <th>
                        Actual Negative
                      </th>

                      <td className="matrix-true-negative">

                        <strong>
                          73
                        </strong>

                        <span>
                          True Negative
                        </span>

                      </td>

                      <td className="matrix-false-positive">

                        <strong>
                          27
                        </strong>

                        <span>
                          False Positive
                        </span>

                      </td>

                    </tr>


                    {/* Actual positive class */}
                    <tr>

                      <th>
                        Actual Positive
                      </th>

                      <td className="matrix-false-negative">

                        <strong>
                          16
                        </strong>

                        <span>
                          False Negative
                        </span>

                      </td>

                      <td className="matrix-true-positive">

                        <strong>
                          38
                        </strong>

                        <span>
                          True Positive
                        </span>

                      </td>

                    </tr>

                  </tbody>

                </table>

              </div>


              {/* Confusion matrix explanation */}
              <div className="research-matrix-summary">

                <p>
                  <strong>True Positive (38):</strong>
                  Actual positive cases correctly predicted as positive.
                </p>

                <p>
                  <strong>True Negative (73):</strong>
                  Actual negative cases correctly predicted as negative.
                </p>

                <p>
                  <strong>False Positive (27):</strong>
                  Actual negative cases incorrectly predicted as positive.
                </p>

                <p>
                  <strong>False Negative (16):</strong>
                  Actual positive cases incorrectly predicted as negative.
                </p>

              </div>

            </section>
            )}



            {/* ==================================================
                DATASET AND EXPERIMENTAL METHODOLOGY
                ================================================== */}

            {activeReportTab === 'methodology' && (
            <section className="dashboard-panel premium-panel research-methodology-panel">

              <div className="research-report-heading">

                <div>

                  <p className="section-eyebrow">
                    EXPERIMENTAL DESIGN
                  </p>

                  <h3>
                    Dataset and Methodology
                  </h3>

                  <p>
                    Summary of the dataset and evaluation procedure.
                  </p>

                </div>

              </div>


              <div className="research-methodology-grid">

                {/* Dataset information */}
                <div className="research-methodology-card">

                  <h4>
                    Dataset Information
                  </h4>

                  <div className="research-detail-row">

                    <span>
                      Dataset
                    </span>

                    <strong>
                      Pima Indians Diabetes
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Total Records
                    </span>

                    <strong>
                      768
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Input Features
                    </span>

                    <strong>
                      8
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Training Records
                    </span>

                    <strong>
                      614
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Test Records
                    </span>

                    <strong>
                      154
                    </strong>

                  </div>

                </div>


                {/* Model and evaluation methodology */}
                <div className="research-methodology-card">

                  <h4>
                    Evaluation Methodology
                  </h4>

                  <div className="research-detail-row">

                    <span>
                      Selected Model
                    </span>

                    <strong>
                      Logistic Regression
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Data Splitting
                    </span>

                    <strong>
                      Stratified Split
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Random State
                    </span>

                    <strong>
                      42
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Training Strategy
                    </span>

                    <strong>
                      Training-Only Tuning
                    </strong>

                  </div>

                  <div className="research-detail-row">

                    <span>
                      Threshold Selection
                    </span>

                    <strong>
                      Training OOF
                    </strong>

                  </div>

                </div>

              </div>

            </section>
            )}



            {/* ==================================================
                RESEARCH FINDINGS
                Interpretation of independent test results.
                ================================================== */}

            {activeReportTab === 'findings' && (
            <section className="dashboard-panel premium-panel research-findings-panel">

              <div className="research-report-heading">

                <div>

                  <p className="section-eyebrow">
                    RESEARCH INTERPRETATION
                  </p>

                  <h3>
                    Key Research Findings
                  </h3>

                  <p>
                    Observations based on the independent test evaluation.
                  </p>

                </div>

              </div>


              <div className="research-findings-list">

                <article className="research-finding-item">

                  <span className="research-finding-number">
                    01
                  </span>

                  <div>

                    <h4>
                      Recall Performance
                    </h4>


                    <p>
                      The model achieved 70.37% recall, identifying
                      38 of the 54 actual positive cases in the
                      independent test set.
                    </p>

                  </div>

                </article>


                <article className="research-finding-item">

                  <span className="research-finding-number">
                    02
                  </span>

                  <div>

                    <h4>
                      False Positive Observations
                    </h4>

                    <p>
                      The model produced 27 false positive predictions.
                      This indicates that some negative cases were
                      incorrectly classified as positive.
                    </p>

                  </div>

                </article>


                <article className="research-finding-item">

                  <span className="research-finding-number">
                    03
                  </span>

                  <div>

                    <h4>
                      Model Discrimination
                    </h4>

                    <p>
                      The ROC-AUC score of 80.85% indicates the model's
                      ability to distinguish between the two classes
                      across different classification thresholds.
                    </p>

                  </div>

                </article>


                <article className="research-finding-item">

                  <span className="research-finding-number">
                    04
                  </span>

                  <div>

                    <h4>
                      Explainable AI Integration
                    </h4>

                    <p>
                      SHAP analysis is integrated to explain how
                      individual input features contribute to
                      model outputs.
                    </p>

                  </div>

                </article>

              </div>

            </section>
            )}



            {/* ==================================================
                RESEARCH LIMITATIONS AND DISCLAIMER
                ================================================== */}

            {activeReportTab === 'limitations' && (
            <section className="research-limitations-panel">

              <div className="research-limitations-heading">

                <span>
                  ⓘ
                </span>

                <h3>
                  Research Limitations
                </h3>

              </div>

              <ul>

                <li>
                  The evaluation uses a relatively small independent
                  test set of 154 records.
                </li>

                <li>
                  The model has not undergone clinical validation
                  or prospective testing.
                </li>

                <li>
                  The threshold selection procedure may introduce
                  selection optimism; nested cross-validation could
                  provide a more rigorous evaluation.
                </li>

                <li>
                  The dataset results may not generalize to every
                  population or healthcare setting.
                </li>

              </ul>

              <p className="research-academic-disclaimer">
                DiaPredict is an academic machine-learning research
                project. Its predictions are not medical diagnoses
                and must not replace professional medical advice.
              </p>

            </section>
            )}



            {/* Research report footer */}
            <footer className="research-report-footer">

              <p>
                DiaPredict AI · Phase 10 Research Evaluation
              </p>

              <p>
                Independent Test Results · 154 Records
              </p>

            </footer>

          </section>

        ) : activePage === 'about' ? (

          <section className="dashboard-content about-project-page">
            <div className="history-page-heading">
              <div>
                <p className="section-eyebrow">DIAPREDICT · PROJECT DOCUMENTATION</p>
                <h2>About Project</h2>
                <p>Explore the project background, objectives, methodology, architecture and research context.</p>
              </div>
            </div>

            <div className="shap-tabs about-tabs" role="tablist" aria-label="About Project sections">
              {[
                ['overview', '1', 'Overview'],
                ['objectives', '2', 'Objectives'],
                ['methodology', '3', 'Methodology'],
                ['architecture', '4', 'Architecture'],
                ['technology', '5', 'Technology Stack'],
                ['team', '6', 'Research & Team'],
                ['limitations', '7', 'Limitations'],
              ].map(([id, number, label]) => (
                <button key={id} type="button" role="tab"
                  aria-selected={activeAboutTab === id}
                  className={`shap-tab ${activeAboutTab === id ? 'active' : ''}`}
                  onClick={() => setActiveAboutTab(id)}>
                  <span className="shap-tab-icon">{number}</span><span>{label}</span>
                </button>
              ))}
            </div>

            <div className="dashboard-panel premium-panel about-tab-panel">
              {activeAboutTab === 'overview' && (
                <>
                  <p className="section-eyebrow">01 · PROJECT OVERVIEW</p>
                  <h3>DiaPredict – Explainable AI-Based Diabetes Risk Prediction</h3>
                  <p>DiaPredict is an academic machine-learning research system that estimates diabetes risk from selected health measurements. It combines a trained Logistic Regression model, a web application, prediction history and SHAP-based explanations to make model behaviour easier to inspect.</p>
                  <p>The project uses the Pima Indians Diabetes Dataset containing 768 records and eight input features. Its outputs are research predictions and must not be treated as a clinical diagnosis.</p>
                </>
              )}
              {activeAboutTab === 'objectives' && (
                <>
                  <p className="section-eyebrow">02 · PROJECT OBJECTIVES</p><h3>Project Objectives</h3>
                  <ul>
                    <li>Develop a machine-learning workflow for diabetes outcome prediction using structured health data.</li>
                    <li>Compare multiple classification algorithms using consistent evaluation metrics.</li>
                    <li>Provide an authenticated interface for entering data and reviewing prediction history.</li>
                    <li>Use SHAP visualizations to explain global and individual model behaviour.</li>
                    <li>Present model performance, dataset characteristics and research limitations transparently.</li>
                  </ul>
                </>
              )}
              {activeAboutTab === 'methodology' && (
                <>
                  <p className="section-eyebrow">03 · METHODOLOGY</p><h3>Research Methodology</h3>
                  <ol>
                    <li><strong>Dataset:</strong> Load and inspect the Pima Indians Diabetes Dataset (768 records, eight predictors and one outcome label).</li>
                    <li><strong>Preprocessing:</strong> Investigate implausible zero measurements, imputation, outlier handling and feature scaling.</li>
                    <li><strong>Model development:</strong> Evaluate Logistic Regression, Decision Tree, Random Forest, KNN and SVM.</li>
                    <li><strong>Evaluation:</strong> Use cross-validation and an independent test set, reporting accuracy, precision, recall, F1-score and ROC-AUC separately.</li>
                    <li><strong>Explainability:</strong> Apply SHAP to examine feature contributions at global and individual levels.</li>
                  </ol>
                </>
              )}
              {activeAboutTab === 'architecture' && (
                <>
                  <p className="section-eyebrow">04 · SYSTEM ARCHITECTURE</p><h3>Application Architecture</h3>
                  <div className="research-methodology-grid">
                    <div className="research-methodology-card"><h4>React Frontend</h4><p>Provides login, registration, dashboard, prediction form, history, Model Insights, SHAP explanations and research reports.</p></div>
                    <div className="research-methodology-card"><h4>FastAPI Backend</h4><p>Handles authenticated API requests, prediction workflows, research insight endpoints and communication with the data layer.</p></div>
                    <div className="research-methodology-card"><h4>Machine Learning Layer</h4><p>Loads the trained model and preprocessing components to generate predictions and explanations.</p></div>
                    <div className="research-methodology-card"><h4>Database & Research Assets</h4><p>Stores user and prediction records; dataset, model and evaluation artifacts support the research workflow.</p></div>
                  </div>
                </>
              )}
              {activeAboutTab === 'technology' && (
                <>
                  <p className="section-eyebrow">05 · TECHNOLOGY STACK</p><h3>Technologies Used</h3>
                  <div className="research-methodology-grid">
                    <div className="research-methodology-card"><h4>Frontend</h4><p>React, JavaScript, Vite, HTML and CSS.</p></div>
                    <div className="research-methodology-card"><h4>Backend</h4><p>Python and FastAPI.</p></div>
                    <div className="research-methodology-card"><h4>Machine Learning</h4><p>Scikit-learn, Pandas, NumPy and SHAP.</p></div>
                    <div className="research-methodology-card"><h4>Data & Development</h4><p>MySQL, SQLAlchemy, JWT authentication, Git and GitHub.</p></div>
                  </div>
                </>
              )}
              {activeAboutTab === 'team' && (
                <>
                  <p className="section-eyebrow">06 · RESEARCH & TEAM</p><h3>Research Contributions</h3>
                  <p>The project extends the earlier minor-project foundation with expanded model evaluation, preprocessing experiments, independent testing, explainability and a full-stack application.</p>
                  <h4>Project Team</h4>
                  <ul><li>Hari Krishna Behera</li><li>Amit Ranjan Sahu</li><li>Ajit Kumar Dash</li><li>Tarun Sagar Pattnaik</li></ul>
                  <p><strong>Project Advisor:</strong> Ashalata Panigrahi</p>
                  <p><strong>Programme:</strong> B.Tech – Computer Science and Engineering</p>
                </>
              )}
              {activeAboutTab === 'limitations' && (
                <>
                  <p className="section-eyebrow">07 · LIMITATIONS & DISCLAIMER</p><h3>Research Limitations</h3>
                  <ul>
                    <li>The dataset is relatively small and may not represent all populations or healthcare settings.</li>
                    <li>Model performance on this dataset does not establish clinical effectiveness.</li>
                    <li>Evaluation results depend on the dataset, split, preprocessing and threshold-selection procedure.</li>
                    <li>External validation and prospective clinical studies would be needed before any real-world medical use.</li>
                  </ul>
                  <p className="research-academic-disclaimer">DiaPredict is an academic research project. Its predictions are not medical diagnoses and must not replace professional medical advice.</p>
                </>
              )}
            </div>
          </section>

        ) : (

          /* ==================================================
             7.9 FUTURE MODULE PLACEHOLDER
             Used for modules not yet implemented.
             ================================================== */

          <section className="dashboard-content">

            <div className="dashboard-panel section-placeholder premium-panel">

              <div className="placeholder-icon">
                ◈
              </div>

              <p className="section-eyebrow">
                DIAPREDICT MODULE
              </p>

              <h2>
                {pageTitle}
              </h2>

              <p>
                This module is scheduled for implementation
                in the frontend development phase.
              </p>

              <button
                type="button"
                className="primary-button"
                onClick={() => setActivePage('dashboard')}
              >
                Return to Dashboard
              </button>

            </div>

          </section>

        )}


      </main>

    </div>

  )

}


// ==========================================================
// 8. EXPORT APPLICATION
// ==========================================================

// Export App so main.jsx can render it using ReactDOM.

export default App