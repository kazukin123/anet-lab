// GridMaze.cpp

#include "anet/env/GridMaze.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "GridMazeEnv.hpp"
#include "GridMazeView.hpp"

void anet::rl::env::InitGridMaze()
{
	anet::rl::RegistEnvFactory(std::make_shared<anet::rl::env::GridMazeEnvFactory>());
	anet::rl::gui::RegisterViewCreator<anet::rl::env::GridMazeViewCreator>();
}

