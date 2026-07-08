module {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<2xf32>, %arg1: tensor<2x2xf32>) -> (tensor<2xf32>, tensor<2xui64>) {
    %0 = stablehlo.cholesky %arg1, lower = true : tensor<2x2xf32>
    %1, %2 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2xui32>)
    %3 = stablehlo.constant dense<9> : tensor<2xui32>
    %4 = stablehlo.shift_right_logical %2, %3 : tensor<2xui32>
    %5 = stablehlo.convert %4 : (tensor<2xui32>) -> tensor<2xf32>
    %6 = stablehlo.constant dense<1.1920929E-7> : tensor<2xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<2xf32>
    %8 = stablehlo.constant dense<2.0> : tensor<2xf32>
    %9 = stablehlo.constant dense<1.0> : tensor<2xf32>
    %10 = stablehlo.multiply %7, %8 : tensor<2xf32>
    %11 = stablehlo.subtract %10, %9 : tensor<2xf32>
    %12 = chlo.erf_inv %11 : tensor<2xf32> -> tensor<2xf32>
    %13 = stablehlo.constant dense<1.4142135> : tensor<2xf32>
    %14 = stablehlo.multiply %12, %13 : tensor<2xf32>
    %15 = stablehlo.dot_general %0, %14, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2x2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %16 = stablehlo.add %arg0, %15 : tensor<2xf32>
    return %16, %1 : tensor<2xf32>, tensor<2xui64>
  }
}
