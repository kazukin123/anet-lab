
set "run="

for /f "delims=" %%A in ('dir logs /b /o:n') do (
    set "run=%%A"
)

echo RUN: %run%

cd "logs\%run%\heatmap_05qtime"
ffmpeg -y -r 30 -i heatmap_05qtime_%%08d.png -vcodec libx264 -pix_fmt yuv420p -r 60 ../../../out.mp4 
start ../../../out.mp4
