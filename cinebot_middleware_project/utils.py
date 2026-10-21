from langgraph.types import Command

def pretty_print_messages(result):
    for i, message in enumerate(result.get("messages", []), 1):
        print(f"\n{'=' * 80}")
        print(f"Message {i}: {message.__class__.__name__}")
        print("=" * 80)

        # Basic message information
        print(f"ID: {getattr(message, 'id', None)}")

        # Message content
        content = getattr(message, "content", "")
        if content:
            print("\nContent:")
            print(content)

        # Tool calls
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("\nTool Calls:")
            for tool in tool_calls:
                print(f"  • {tool['name']}")
                print(f"    Args: {tool['args']}")
                print(f"    ID:   {tool['id']}")

        # Tool message information
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            print(f"\nTool Call ID: {tool_call_id}")

        # Summarization information
        additional_kwargs = getattr(message, "additional_kwargs", {})
        if additional_kwargs.get("lc_source"):
            print(f"\nSource: {additional_kwargs['lc_source']}")

    print(f"\n{'=' * 80}")
    print("END OF MESSAGE HISTORY")
    print("=" * 80)

def run_interactive_hitl_demo(agent, config):
    """A genuinely interactive HITL loop -- ask out loud, type the answer, watch it apply live."""
    state = agent.get_state(config)
    if not state.next:
        print("Nothing is currently paused for approval.")
        return

    print("The agent wants to call a guarded tool. Choose a decision:")
    print("  1) approve  -- run it exactly as proposed")
    print("  2) edit     -- run it, but change the booking_id first")
    print("  3) reject   -- block it, with a reason sent back to the agent")
    print("  4) respond  -- answer a question instead of deciding on the action")

    choice = input("Type 1, 2, 3, or 4: ").strip()

    if choice == "1":
        decision = {"type": "approve"}
    elif choice == "2":
        new_id = input("New booking_id to use instead: ").strip()
        decision = {"type": "edit", "args": {"booking_id": new_id}}
    elif choice == "3":
        reason = input("Reason for rejecting: ").strip()
        decision = {"type": "reject", "message": reason}
    elif choice == "4":
        answer = input("Your response to the agent: ").strip()
        decision = {"type": "respond", "message": answer}
    else:
        print("Not a valid choice -- try again.")
        return

    resumed = agent.invoke(Command(resume={"decisions": [decision]}), config=config)
    print()
    print("Agent's final response:", resumed["messages"][-1].content)

     
