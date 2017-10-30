rem TODO: add version and edition autodetection and merge into make.bat

mkdir obj

CD zstd-1.3.0\build\VS_scripts

CALL .\build.VS2017.cmd
COPY .\bin\Release\x64\libzstd_static.lib ..\..\..\obj

CD ..\..\..

SET PATH=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin;%PATH%

SET "VSCMD_START_DIR=%CD%"
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

cl.exe /std:c++14 /O2 /MT /EHsc /I . /Fo.\obj\ ^
 .\obj\libzstd_static.lib ^
 .\main.cpp ^
 .\core\Constants.cpp .\core\Entity.cpp .\core\GameEvent.cpp .\core\Halite.cpp .\core\hlt.cpp .\core\SimulationEvent.cpp ^
 .\core\Replay.cpp .\core\Statistics.cpp ^
 .\networking\Networking.cpp .\networking\BotInputError.cpp ^
 .\core\mapgen\Generator.cpp .\core\mapgen\SolarSystem.cpp .\core\mapgen\AsteroidCluster.cpp ^
 /link /ltcg /implib:obj\halite.lib /out:halite.exe
