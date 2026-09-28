# examples/eco65_sim/Dockerfile
# Extends the LIBERO runtime image with ECO65 support.
# Build: docker build -t eco65_runtime -f examples/eco65_sim/Dockerfile .

FROM libero

# Mount robosuite with ECO65 (PYTHONPATH=/app ensures local robosuite takes precedence)
# The project root is mounted at /app by compose.yml, giving access to /app/robosuite/
# and /app/examples/eco65_sim/

CMD ["/bin/bash", "-c", "source /.venv/bin/activate && python examples/eco65_sim/main.py $CLIENT_ARGS"]
