#include "fake_bot.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char * argv[]) {

    // generate a unique ID via current nanosecond time
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    const int n = snprintf(NULL, 0, "%ld", ts.tv_nsec); // length of nsec field without null byte
    assert(n > 0);
    char timeId[n + 1];
    int c = snprintf(timeId, n + 1, "%lu", ts.tv_nsec);
    assert(timeId[n] == '\0');
    assert(c == n);

    int fifoNameSz = strlen(NAMED_PIPE_PREFIX) + n + 1;
    char baseFifoName[fifoNameSz];
    memset(baseFifoName, 0, fifoNameSz);
    snprintf(baseFifoName, fifoNameSz, "%s%s", NAMED_PIPE_PREFIX, timeId);

    char toFifoName[fifoNameSz + strlen(TO_HALITE_SUFFIX) + 1];
    char fromFifoName[fifoNameSz + strlen(FROM_HALITE_SUFFIX) + 1];

    snprintf(toFifoName, fifoNameSz + strlen(TO_HALITE_SUFFIX) + 1, "%s%s", baseFifoName, TO_HALITE_SUFFIX);
    snprintf(fromFifoName, fifoNameSz + strlen(FROM_HALITE_SUFFIX) + 1, "%s%s", baseFifoName, FROM_HALITE_SUFFIX);

    // create 2 named pipes, 1 for reading and 1 for writing
    if (mkfifo(toFifoName, 0666) != 0) {
        perror("to pipe");
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "Created fifo to halite: %s%s\n", toFifoName);

    if (mkfifo(fromFifoName, 0666) != 0) {
        perror("from pipe");
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "Created fifo from halite: %s%s\n", fromFifoName);

    // open named pipes
    int fromPipeFd = open(fromFifoName, O_RDONLY);
    int toPipeFd = open(toFifoName, O_WRONLY);

    if (fromPipeFd < 0) {
        perror("from pipe open");
        exit(EXIT_FAILURE);
    }

    if (toPipeFd < 0) {
        perror("to pipe open");
        exit(EXIT_FAILURE);
    }

    // stdin to this program goes to from_halite pipe
    if (dup2(fromPipeFd, STDIN_FILENO) < 0) {
        perror("redirect stdin");
        exit(EXIT_FAILURE);
    }

    close(fromPipeFd);

    // to_halite pipe is redirected to stdout of this program
    if (dup2(toPipeFd, STDOUT_FILENO) < 0) {
        perror("redirect stdout");
        exit(EXIT_FAILURE);
    }

    close(toPipeFd);

    return EXIT_SUCCESS;

}
