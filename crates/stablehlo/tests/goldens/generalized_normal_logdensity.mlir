module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg1, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.compare LE, %arg2, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %3 = stablehlo.select %1, %1, %2 : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = stablehlo.constant dense<0.5> : tensor<f32>
    %8 = stablehlo.log %arg2 : tensor<f32>
    %9 = stablehlo.constant dense<2.0> : tensor<f32>
    %10 = stablehlo.multiply %9, %arg1 : tensor<f32>
    %11 = stablehlo.log %10 : tensor<f32>
    %12 = stablehlo.negate %11 : tensor<f32>
    %13 = stablehlo.constant dense<1.0> : tensor<f32>
    %14 = stablehlo.divide %13, %arg2 : tensor<f32>
    %15 = chlo.lgamma %14 : tensor<f32> -> tensor<f32>
    %16 = stablehlo.negate %15 : tensor<f32>
    %17 = stablehlo.subtract %7, %arg0 : tensor<f32>
    %18 = stablehlo.abs %17 : tensor<f32>
    %19 = stablehlo.divide %18, %arg1 : tensor<f32>
    %20 = stablehlo.power %19, %arg2 : tensor<f32>
    %21 = stablehlo.negate %20 : tensor<f32>
    %22 = stablehlo.add %8, %12 : tensor<f32>
    %23 = stablehlo.add %22, %16 : tensor<f32>
    %24 = stablehlo.add %23, %21 : tensor<f32>
    return %24, %6 : tensor<f32>, tensor<i32>
  }
}
