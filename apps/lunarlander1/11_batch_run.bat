@echo off
SET EXE="bin\Release\LunarLanderRLGUI.exe"

call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=8680553529747499138"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=9527375268760563797"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=7159038980716138760"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=17268299675072012765"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=15840880607899339877"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=45934658819433618"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=3512965569908297564"
call:run_exe "LunarLanderApp.$=LunarLanderApp.batchrun" "train.seed=2794877750456487840"

pause
exit /b


:run_exe
echo %DATE% %TIME% START %1 %2 %3 %4
%EXE% %1 %2 %3 %4
echo %DATE% %TIME% END   %1 %2 %3 %4
exit /b
