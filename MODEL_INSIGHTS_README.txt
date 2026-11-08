DiaPredict - Model Insights module update
=========================================

This patch contains only the new module files and the two integration edits.
Extract this ZIP into the ROOT of your existing diabetes-prediction-major project
(the folder that contains frontend/, backend/, data/, models/, src/, and results/).
Allow Windows to overwrite the existing App.jsx and main.py when prompted.

Files:
- frontend/src/App.jsx                         (updated: Model Insights route/import)
- frontend/src/ModelInsights.jsx               (new React page)
- frontend/src/ModelInsights.css               (new styles)
- backend/app/main.py                          (updated: registers insights router)
- backend/app/api/insights.py                  (new authenticated API endpoint)
- backend/app/services/insights_service.py    (new real-data analytics service)

Backend endpoint:
GET http://127.0.0.1:8000/insights/overview
Requires the same JWT Bearer token used by the other authenticated endpoints.

Important:
- Keep data/diabetes.csv and results/*.csv in their existing project-root locations.
- Do not upload or share your .env file.
- No database migration is required for this read-only insights endpoint.
- The page reads dataset statistics and research metrics from existing CSV artifacts.
- The charts do not use fabricated placeholder values.

After extracting:
1. Start FastAPI from the project root using your existing backend command.
2. Start the frontend using npm run dev.
3. Log in and open Model Insights in the sidebar.
4. If the API returns an error, send the exact browser error and backend terminal traceback.

Validation note:
The new Python service was executed against the supplied data/results and returned
768 records, 5 model-comparison rows, 8 correlation rows, and the independent-test
metrics. A Vite production build could not be completed in this Linux inspection
environment because the uploaded node_modules contains Windows-only native
Rolldown binaries; run npm run build on your Windows machine.
