#!/bin/bash

set -euo pipefail


SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

STDLIB="${SHAREMIND_PREFIX_PATH}/lib/sharemind/stdlib"


compile() {
    name=$(basename $1 .sc)
    sc_dir=$(dirname -- $1)
    sb="${name}.sb"
    sa="${name}.sa"

    echo "Compiling $1 to $sb"

    scc \
        -I $SCRIPT_PATH \
        -I $STDLIB \
        -I $sc_dir \
        --input $1 \
        --output $SCRIPT_PATH/bytecode/$sb

}


if [ $# -eq 0 ]; then
    echo "No secrec files given"
else
    for var in "$@"; do
        compile $var
    done
fi


