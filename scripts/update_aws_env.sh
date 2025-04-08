#!/bin/bash

# Script to fetch temporary AWS credentials and update a .env file.

# --- Configuration ---
ENV_FILE=".env" # The name of the .env file to update in the current directory
DURATION_SECONDS=43200 # Duration for the temporary credentials (e.g., 12 hours = 43200)
PROFILE_NAME="" # Optional: Specify an AWS CLI profile name if not using default
# Example: PROFILE_NAME="my-dev-profile"

# --- Check Prerequisites ---
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed or not found in PATH."
    echo "Please install and configure it first."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed or not found in PATH."
    echo "Please install jq (e.g., 'sudo apt install jq' or 'brew install jq')."
    exit 1
fi

# --- Construct AWS CLI Command ---
AWS_CMD="aws sts get-session-token --duration-seconds ${DURATION_SECONDS}"
if [[ -n "${PROFILE_NAME}" ]]; then
    AWS_CMD="${AWS_CMD} --profile ${PROFILE_NAME}"
fi

# --- Fetch Credentials ---
echo "Fetching temporary AWS credentials..."
CREDENTIALS_JSON=$(eval ${AWS_CMD}) # Use eval to handle optional profile flag correctly

# Check if the AWS command was successful
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to fetch AWS credentials. Check your AWS CLI configuration and permissions."
    echo "Command attempted: ${AWS_CMD}"
    # Optionally print the error output from AWS CLI if needed for debugging
    # echo "AWS CLI Output:"
    # echo "${CREDENTIALS_JSON}" # This variable might contain error details
    exit 1
fi

# --- Parse Credentials using jq ---
ACCESS_KEY_ID=$(echo "${CREDENTIALS_JSON}" | jq -r '.Credentials.AccessKeyId')
SECRET_ACCESS_KEY=$(echo "${CREDENTIALS_JSON}" | jq -r '.Credentials.SecretAccessKey')
SESSION_TOKEN=$(echo "${CREDENTIALS_JSON}" | jq -r '.Credentials.SessionToken')
EXPIRATION=$(echo "${CREDENTIALS_JSON}" | jq -r '.Credentials.Expiration')

# Check if parsing was successful
if [[ -z "${ACCESS_KEY_ID}" || "${ACCESS_KEY_ID}" == "null" || \
      -z "${SECRET_ACCESS_KEY}" || "${SECRET_ACCESS_KEY}" == "null" || \
      -z "${SESSION_TOKEN}" || "${SESSION_TOKEN}" == "null" ]]; then
    echo "Error: Failed to parse credentials from AWS CLI output."
    echo "AWS CLI Output:"
    echo "${CREDENTIALS_JSON}"
    exit 1
fi

echo "Successfully fetched credentials. Expiration: ${EXPIRATION}"

# --- Update .env File ---
TEMP_ENV_FILE=$(mktemp) # Create a temporary file

# Check if .env file exists and filter out old AWS keys AND the old comment line
if [[ -f "${ENV_FILE}" ]]; then
    echo "Updating existing ${ENV_FILE}..."
    # Use grep -vE to exclude lines starting with the AWS keys OR the specific comment we add
    grep -vE '^AWS_ACCESS_KEY_ID=|^AWS_SECRET_ACCESS_KEY=|^AWS_SESSION_TOKEN=|^# AWS Credentials fetched at' "${ENV_FILE}" > "${TEMP_ENV_FILE}"
else
    echo "Creating new ${ENV_FILE}..."
    # If file doesn't exist, the temp file will be empty initially
fi

# Append the new credentials to the temporary file
# Adding quotes around values is generally safer for .env files
echo "" >> "${TEMP_ENV_FILE}" # Add a newline for separation
echo "# AWS Credentials fetched at $(date) - Expires: ${EXPIRATION}" >> "${TEMP_ENV_FILE}"
echo "AWS_ACCESS_KEY_ID=\"${ACCESS_KEY_ID}\"" >> "${TEMP_ENV_FILE}"
echo "AWS_SECRET_ACCESS_KEY=\"${SECRET_ACCESS_KEY}\"" >> "${TEMP_ENV_FILE}"
echo "AWS_SESSION_TOKEN=\"${SESSION_TOKEN}\"" >> "${TEMP_ENV_FILE}"

# Replace the original .env file with the updated temporary file
mv "${TEMP_ENV_FILE}" "${ENV_FILE}"

# Ensure the temporary file is removed even if mv fails (though unlikely)
rm -f "${TEMP_ENV_FILE}"

echo "${ENV_FILE} updated successfully."
exit 0

