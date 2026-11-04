module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LT, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<"0x00004040"> : tensor<f32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.compare EQ, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %10 = stablehlo.select %9, %8, %arg0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %15 = stablehlo.log %10 : tensor<f32>
    %52 = stablehlo.multiply %6, %15 : tensor<f32>
    %54 = stablehlo.constant dense<"0x6058E53F"> : tensor<f32>
    %55 = stablehlo.subtract %52, %10 : tensor<f32>
    %56 = stablehlo.subtract %55, %54 : tensor<f32>
    %99 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %100 = stablehlo.negate %99 : tensor<f32>
    %101 = stablehlo.select %9, %100, %56 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    return %101, %4 : tensor<f32>, tensor<i32>
  }
}
