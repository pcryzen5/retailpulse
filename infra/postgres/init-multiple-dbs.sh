#!/usr/bin/env bash
set -e
set -u

echo ">>> Initializing RetailPulse multi-database environment..."

RP_OWNER_PASS="${RP_DB_OWNER_PASSWORD:-devownerpassword}"
RP_APP_PASS="${RP_DB_APP_PASSWORD:-devpassword}"
MLFLOW_PASS="${MLFLOW_DB_PASSWORD:-mlflowpassword}"
PREFECT_PASS="${PREFECT_DB_PASSWORD:-prefectpassword}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create application roles if they do not exist
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rp_owner') THEN
            CREATE ROLE rp_owner LOGIN PASSWORD '$RP_OWNER_PASS';
        END IF;
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rp_app') THEN
            CREATE ROLE rp_app LOGIN PASSWORD '$RP_APP_PASS';
        END IF;
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mlflow') THEN
            CREATE ROLE mlflow LOGIN PASSWORD '$MLFLOW_PASS';
        END IF;
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'prefect') THEN
            CREATE ROLE prefect LOGIN PASSWORD '$PREFECT_PASS';
        END IF;
    END
    \$\$;

    -- Create databases with respective owners
    SELECT 'CREATE DATABASE retailpulse OWNER rp_owner'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'retailpulse')\gexec

    SELECT 'CREATE DATABASE mlflow OWNER mlflow'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

    SELECT 'CREATE DATABASE prefect OWNER prefect'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prefect')\gexec
EOSQL

echo ">>> Configuring schemas and privileges on retailpulse database..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "retailpulse" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS citext;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION rp_owner;
    CREATE SCHEMA IF NOT EXISTS ops AUTHORIZATION rp_owner;
    CREATE SCHEMA IF NOT EXISTS core AUTHORIZATION rp_owner;
    CREATE SCHEMA IF NOT EXISTS ml AUTHORIZATION rp_owner;

    GRANT USAGE ON SCHEMA app, ops, core, ml TO rp_app;
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA app, ops, core, ml TO rp_app;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA app, ops, core, ml TO rp_app;

    ALTER DEFAULT PRIVILEGES FOR ROLE rp_owner IN SCHEMA app, ops, core, ml
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rp_app;
    ALTER DEFAULT PRIVILEGES FOR ROLE rp_owner IN SCHEMA app, ops, core, ml
        GRANT USAGE, SELECT ON SEQUENCES TO rp_app;
EOSQL

echo ">>> Database initialization complete."
