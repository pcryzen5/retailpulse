// k6 Load Test Scenario for RetailPulse API
// Implements Task P9-02: 50 concurrent users scenario
// Usage: k6 run tests/load/k6_scenario.js

import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '10s', target: 50 }, // Ramp-up to 50 concurrent virtual users
    { duration: '40s', target: 50 }, // Steady state at 50 concurrent users
    { duration: '10s', target: 0 },  // Graceful ramp-down
  ],
  thresholds: {
    // TRD §7 SLA: p95 < 1000 ms for cached KPI aggregates
    'http_req_duration{endpoint:kpis_overview}': ['p(95)<1000'],
    // TRD §7 SLA: p95 < 300 ms for indexed read endpoints
    'http_req_duration{endpoint:forecasts}': ['p(95)<300'],
    'http_req_duration{endpoint:customers}': ['p(95)<300'],
    'http_req_duration{endpoint:inventory}': ['p(95)<300'],
    // Overall reliability
    'http_req_failed': ['rate<0.01'], // Failure rate must be < 1%
  },
};

const BASE_URL = __ENV.RP_API_URL || 'http://localhost:8000';
const TOKEN = __ENV.RP_BEARER_TOKEN || '';

const params = {
  headers: {
    'Accept': 'application/json',
    ...(TOKEN ? { 'Authorization': `Bearer ${TOKEN}` } : {}),
  },
};

export default function () {
  // 1. KPI Overview (cached aggregate)
  const kpiRes = http.get(`${BASE_URL}/api/v1/kpis/overview`, {
    ...params,
    tags: { endpoint: 'kpis_overview' },
  });
  check(kpiRes, { 'status is 200': (r) => r.status === 200 });

  sleep(0.2);

  // 2. Demand Forecast (indexed read)
  const productId = Math.floor(Math.random() * 100) + 1;
  const forecastRes = http.get(`${BASE_URL}/api/v1/forecasts?product_id=${productId}&store_id=1&horizon=28`, {
    ...params,
    tags: { endpoint: 'forecasts' },
  });
  check(forecastRes, { 'status is 200': (r) => r.status === 200 });

  sleep(0.2);

  // 3. Customer Churn Risk List (indexed read)
  const custRes = http.get(`${BASE_URL}/api/v1/customers?risk_band=high&page=1&page_size=20`, {
    ...params,
    tags: { endpoint: 'customers' },
  });
  check(custRes, { 'status is 200': (r) => r.status === 200 });

  sleep(0.2);

  // 4. Inventory Recommendations (indexed read)
  const invRes = http.get(`${BASE_URL}/api/v1/inventory/recommendations?status=reorder_now&page=1&page_size=20`, {
    ...params,
    tags: { endpoint: 'inventory' },
  });
  check(invRes, { 'status is 200': (r) => r.status === 200 });

  sleep(0.3);
}
