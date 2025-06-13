import pytest
from unittest.mock import patch, MagicMock
import sys
from langchain_core.messages import HumanMessage, AIMessage


@patch('my_agent.utils.tools.TavilySearchResults')
class TestAgent:
    def test_graph_config(self, mock_tavily):
        """Test that GraphConfig accepts valid model names"""
        mock_tavily.return_value = MagicMock()
        from my_agent.agent import GraphConfig
        config = GraphConfig(model_name="openrouter")
        assert config["model_name"] == "openrouter"

    @patch('my_agent.utils.nodes.ChatOpenAI')
    def test_get_model_openrouter(self, mock_chat_openai, mock_tavily):
        """Test that _get_model returns correct model for openrouter"""
        mock_tavily.return_value = MagicMock()
        from my_agent.utils.nodes import _get_model
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        
        result = _get_model("openrouter")
        
        mock_chat_openai.assert_called_once()
        mock_model.bind_tools.assert_called_once()

    @patch('my_agent.utils.nodes.ChatAnthropic')
    def test_get_model_anthropic(self, mock_chat_anthropic, mock_tavily):
        """Test that _get_model returns correct model for anthropic"""
        mock_tavily.return_value = MagicMock()
        from my_agent.utils.nodes import _get_model
        mock_model = MagicMock()
        mock_chat_anthropic.return_value = mock_model
        
        result = _get_model("anthropic")
        
        mock_chat_anthropic.assert_called_once()
        mock_model.bind_tools.assert_called_once()

    @patch('my_agent.utils.nodes.ChatOpenAI')
    def test_get_model_openai(self, mock_chat_openai, mock_tavily):
        """Test that _get_model returns correct model for openai"""
        mock_tavily.return_value = MagicMock()
        from my_agent.utils.nodes import _get_model
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model
        
        result = _get_model("openai")
        
        mock_chat_openai.assert_called_once()
        mock_model.bind_tools.assert_called_once()

    def test_get_model_invalid(self, mock_tavily):
        """Test that _get_model raises ValueError for invalid model name"""
        mock_tavily.return_value = MagicMock()
        from my_agent.utils.nodes import _get_model
        with pytest.raises(ValueError, match="Unsupported model type"):
            _get_model("invalid_model")

    @patch('my_agent.utils.nodes._get_model')
    def test_call_model(self, mock_get_model, mock_tavily):
        """Test call_model function"""
        mock_tavily.return_value = MagicMock()
        from my_agent.utils.nodes import call_model
        from my_agent.utils.state import AgentState
        mock_model = MagicMock()
        mock_response = AIMessage(content="Test response")
        mock_model.invoke.return_value = mock_response
        mock_get_model.return_value = mock_model
        
        state = AgentState(messages=[HumanMessage(content="Hello")])
        config = {"configurable": {"model_name": "openrouter"}}
        
        result = call_model(state, config)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0] == mock_response
        mock_get_model.assert_called_once_with("openrouter")

    @patch('my_agent.utils.nodes._get_model')
    def test_call_model_default_config(self, mock_get_model, mock_tavily):
        """Test call_model with default configuration"""
        mock_tavily.return_value = MagicMock()
        from my_agent.utils.nodes import call_model
        from my_agent.utils.state import AgentState
        mock_model = MagicMock()
        mock_response = AIMessage(content="Test response")
        mock_model.invoke.return_value = mock_response
        mock_get_model.return_value = mock_model
        
        state = AgentState(messages=[HumanMessage(content="Hello")])
        config = {}
        
        result = call_model(state, config)
        
        mock_get_model.assert_called_once_with("openrouter")

    def test_agent_state_structure(self, mock_tavily):
        """Test AgentState structure"""
        mock_tavily.return_value = MagicMock()
        from my_agent.utils.state import AgentState
        human_msg = HumanMessage(content="Hello")
        ai_msg = AIMessage(content="Hi there!")
        
        state = AgentState(messages=[human_msg, ai_msg])
        
        assert len(state["messages"]) == 2
        assert state["messages"][0].content == "Hello"
        assert state["messages"][1].content == "Hi there!"

    @patch('my_agent.utils.nodes._get_model')
    def test_graph_execution(self, mock_get_model, mock_tavily):
        """Test basic graph execution"""
        mock_tavily.return_value = MagicMock()
        from my_agent.agent import graph
        mock_model = MagicMock()
        mock_response = AIMessage(content="Test response", tool_calls=[])
        mock_model.invoke.return_value = mock_response
        mock_get_model.return_value = mock_model
        
        config = {"configurable": {"model_name": "openrouter"}}
        initial_state = {"messages": [HumanMessage(content="Hello")]}
        
        result = graph.invoke(initial_state, config)
        
        assert "messages" in result
        assert len(result["messages"]) >= 2  # At least initial message + response


@patch('my_agent.utils.tools.TavilySearchResults')
class TestTools:
    def test_tools_import(self, mock_tavily):
        """Test that tools can be imported correctly"""
        mock_tool = MagicMock()
        mock_tool.name = "tavily_search"
        mock_tavily.return_value = mock_tool
        
        from my_agent.utils.tools import tools
        assert len(tools) > 0


class TestUtilityFunctions:
    """Test utility functions without dependencies"""
    
    def test_human_message_creation(self):
        """Test creating HumanMessage"""
        msg = HumanMessage(content="Hello, world!")
        assert msg.content == "Hello, world!"
        assert msg.type == "human"
    
    def test_ai_message_creation(self):
        """Test creating AIMessage"""
        msg = AIMessage(content="Hi there!")
        assert msg.content == "Hi there!"
        assert msg.type == "ai"