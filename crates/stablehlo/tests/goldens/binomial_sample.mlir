module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<i32>, tensor<2xui64>) {
    %1 = stablehlo.constant dense<0.3> : tensor<f32>
    %2, %3 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<5xui32>)
    %4 = stablehlo.constant dense<9> : tensor<5xui32>
    %5 = stablehlo.shift_right_logical %3, %4 : tensor<5xui32>
    %6 = stablehlo.convert %5 : (tensor<5xui32>) -> tensor<5xf32>
    %7 = stablehlo.constant dense<1.1920929E-7> : tensor<5xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<5xf32>
    %9 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<5xf32>
    %10 = stablehlo.compare LT, %8, %9 : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xi1>
    %11 = stablehlo.constant dense<1.0> : tensor<5xf32>
    %12 = stablehlo.constant dense<0.0> : tensor<5xf32>
    %13 = stablehlo.select %10, %11, %12 : (tensor<5xi1>, tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.reduce(%13 init: %14) applies stablehlo.add across dimensions = [0] : (tensor<5xf32>, tensor<f32>) -> tensor<f32>
    %16 = stablehlo.convert %15 : (tensor<f32>) -> tensor<i32>
    return %16, %2 : tensor<i32>, tensor<2xui64>
  }
}
