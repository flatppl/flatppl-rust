module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.constant dense<0.5> : tensor<f32>
    %8 = stablehlo.divide %5, %arg0 : tensor<f32>
    %9 = stablehlo.log %8 : tensor<f32>
    %10 = stablehlo.log %arg0 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.multiply %8, %8 : tensor<f32>
    %13 = stablehlo.multiply %5, %12 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.add %9, %11 : tensor<f32>
    %16 = stablehlo.add %15, %14 : tensor<f32>
    return %16, %4 : tensor<f32>, tensor<i32>
  }
}
