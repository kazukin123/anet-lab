// GridMaze.cpp

#include "anet/env/ImageCls.hpp"
#include "anet/env.hpp"
#include "anet/gui.hpp"
#include "ImageClsEnv.hpp"
#include "ImageClsView.hpp"

void anet::rl::env::InitImageCls()
{
	anet::rl::RegistEnvFactory(std::make_shared<anet::rl::env::ImageClsEnvFactory>());
	anet::rl::gui::RegisterViewCreator<anet::rl::env::ImageClsViewCreator>();
}
