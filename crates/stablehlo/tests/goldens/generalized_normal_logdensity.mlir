module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.minimum %arg1, %arg2 : tensor<f32>
    %1 = stablehlo.constant dense<0.0> : tensor<f32>
    %2 = stablehlo.compare LE, %0, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<9> : tensor<i32>
    %5 = stablehlo.select %2, %4, %3 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.log %arg2 : tensor<f32>
    %8 = stablehlo.constant dense<2.0> : tensor<f32>
    %9 = stablehlo.multiply %8, %arg1 : tensor<f32>
    %10 = stablehlo.log %9 : tensor<f32>
    %11 = stablehlo.negate %10 : tensor<f32>
    %12 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.divide %12, %arg2 : tensor<f32>
    %14 = chlo.lgamma %13 : tensor<f32> -> tensor<f32>
    %15 = stablehlo.negate %14 : tensor<f32>
    %16 = stablehlo.subtract %6, %arg0 : tensor<f32>
    %17 = stablehlo.abs %16 : tensor<f32>
    %18 = stablehlo.divide %17, %arg1 : tensor<f32>
    %19 = stablehlo.power %18, %arg2 : tensor<f32>
    %20 = stablehlo.negate %19 : tensor<f32>
    %21 = stablehlo.add %7, %11 : tensor<f32>
    %22 = stablehlo.add %21, %15 : tensor<f32>
    %23 = stablehlo.add %22, %20 : tensor<f32>
    return %23, %5 : tensor<f32>, tensor<i32>
  }
}
