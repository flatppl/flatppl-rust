module attributes {flatppl.runtime_status = 1 : i64} {
  func.func @logdensity(%arg0: tensor<f32>) -> (tensor<f32>, tensor<i32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.compare LE, %arg0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = stablehlo.constant dense<0> : tensor<i32>
    %3 = stablehlo.constant dense<9> : tensor<i32>
    %4 = stablehlo.select %1, %3, %2 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = stablehlo.constant dense<0.5> : tensor<f32>
    %6 = stablehlo.constant dense<1.0> : tensor<f32>
    %7 = stablehlo.add %arg0, %6 : tensor<f32>
    %8 = stablehlo.multiply %5, %7 : tensor<f32>
    %9 = chlo.lgamma %8 : tensor<f32> -> tensor<f32>
    %10 = stablehlo.constant dense<3.141592653589793> : tensor<f32>
    %11 = stablehlo.multiply %arg0, %10 : tensor<f32>
    %12 = stablehlo.log %11 : tensor<f32>
    %13 = stablehlo.multiply %5, %12 : tensor<f32>
    %14 = stablehlo.negate %13 : tensor<f32>
    %15 = stablehlo.multiply %5, %arg0 : tensor<f32>
    %16 = chlo.lgamma %15 : tensor<f32> -> tensor<f32>
    %17 = stablehlo.negate %16 : tensor<f32>
    %18 = stablehlo.constant dense<"0x0000803E"> : tensor<f32>
    %19 = stablehlo.divide %18, %arg0 : tensor<f32>
    %20 = stablehlo.add %6, %19 : tensor<f32>
    %21 = stablehlo.log %20 : tensor<f32>
    %22 = stablehlo.multiply %8, %21 : tensor<f32>
    %23 = stablehlo.negate %22 : tensor<f32>
    %24 = stablehlo.add %9, %14 : tensor<f32>
    %25 = stablehlo.add %24, %17 : tensor<f32>
    %26 = stablehlo.add %25, %23 : tensor<f32>
    return %26, %4 : tensor<f32>, tensor<i32>
  }
}
