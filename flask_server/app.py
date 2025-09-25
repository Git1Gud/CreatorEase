from flask import Flask
from flask_cors import CORS

from config import ensure_directories, flask_config
from routes import ALL_BLUEPRINTS


def create_app() -> Flask:
    """Application factory wiring configuration and route blueprints."""

    app = Flask(__name__)
    CORS(app)

    ensure_directories()
    app.config.update(flask_config())

    for blueprint in ALL_BLUEPRINTS:
        app.register_blueprint(blueprint)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)