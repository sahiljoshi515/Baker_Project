"""
This file serves as the controller in our MVC architecture. It is responsible for
listening for user-driven front-end events and updating the front-end
as well as updating information in the backend and listening for SSE's (to be added).
"""

import uvicorn

from app_factory import create_app
from core.config import get_settings
import logging
import logging.config

# Start FastAPI backend using factory pattern with uvicorn to make testing easier (i.e. allows 
# us to inject backends with different middleware/lifespan functions/etc. at startup time)
app = create_app()

# 
#  Main entry point - sets up the app and wires everything together
# 
if __name__ == "__main__":
    settings = get_settings()

    logging.config.fileConfig('..logging.conf')
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
