from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from .graph import app

fastapi_app = FastAPI()

@fastapi_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_request = await websocket.receive_text()
            initial_state = {
                "user_request": user_request,
                "clarification_needed": False,
                "clarification_question": "",
                "clarified_request": "",
                "brief": "",
                "plan": [],
                "queries": [],
                "results": "",
                "all_results": [],
                "report": "",
                "current_task_index": 0
            }
            async for event in app.astream_events(initial_state, version="v1"):
                if event["event"] == "on_chain_end":
                    node = event["name"]
                    output = event["output"]
                    if node == "user_clarification":
                        if output.get("clarification_needed"):
                            await websocket.send_json({"type": "clarification_request", "data": output["clarification_question"]})
                            # Wait for user clarification response
                            clarified_request = await websocket.receive_text()
                            event["state"]["clarified_request"] = clarified_request
                        else:
                            await websocket.send_json({"type": "clarified_request", "data": output.get("clarified_request", "")})
                    elif node == "brief_generation":
                        await websocket.send_json({"type": "brief", "data": output.get("brief", "")})
                    elif node == "supervisor":
                        await websocket.send_json({"type": "plan", "data": output})
                    elif node == "research":
                        await websocket.send_json({"type": "research_result", "data": output})
                    elif node == "writer":
                        await websocket.send_json({"type": "final_report", "data": output})
    except WebSocketDisconnect:
        pass
